"""Runs SimpleTES on config.task's seed program with config.model.name inside vLLM, in synchronous steps.

Every one of config.L steps samples each of config.C chains once. SimpleTES's balance policy picks up to config.N of the
chain's programs, the task's prompt holds them with their scores, and the model answers it config.K times, thinking at
config.model.reasoning_effort when config.model.enable_thinking. The task's evaluator scores every program, then the
policy appends each chain's best to the chain. Failure patterns and reflections are left out.

A response's program is saved to config.output_dir as circle-{l}-{chain_idx}-{k}.py. A response without an EVOLVE-BLOCK
or a program the evaluator rejects is dropped. Each step logs every reward and program to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/simpletes-baseline.py --config config/simpletes-baseline.py
"""

import asyncio
import itertools
import math
import os
import random
import sys
import uuid

import wandb
from absl import flags
from ml_collections import config_flags
from simpletes.node import Node, NodeDatabase, Status
from simpletes.policies.balance import BalancePolicy

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/simpletes-baseline.py", "SimpleTES configuration.")


class Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)
        random.seed(config.seed)
        self.db = NodeDatabase()
        self.selector = BalancePolicy(
            num_chains=config.C,
            max_generations=config.L * config.C * config.K,
            k=config.K,
        )
        os.makedirs(config.output_dir, exist_ok=True)

    def run(self):
        """Runs config.L steps of SimpleTES from the seed program."""
        task, L, C, K, N = self.task, self.config.L, self.config.C, self.config.K, self.config.N
        cfg = self.config.model
        reward = task.evaluate([task.init_program]).item()
        self.db.add(Node(id=uuid.uuid4().hex, code=task.init_program, metrics={"combined_score": reward}, score=reward, status=Status.DONE))
        for l in range(L):
            generation_tasks = []
            for _ in range(C):
                inspirations, _, chain_idx = self.selector.select(self.db, N, stats=None, eval_concurrency=0, backpressure_multiplier=0)
                generation_task = task.generator.create_task_with_gen_id(
                    prompt=task.prompt([node.code for node in inspirations], [node.score for node in inspirations]),
                    inspiration_ids=[node.id for node in inspirations],
                    k=K,
                    chain_idx=chain_idx,
                    gen_id=task.generator.next_gen_id(),
                )
                self.selector.register_batch(generation_task.gen_id, chain_idx, generation_task.inspiration_ids, K)
                generation_tasks.append(generation_task)
            prompt_tokens = self.pipeline.texts_to_tokens([generation_task.prompt for generation_task in generation_tasks], enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort)
            responses = self.pipeline.generate([tokens for tokens in prompt_tokens for _ in range(K)], enable_thinking=cfg.enable_thinking).texts
            rewards = task.evaluate(responses)

            programs = []
            for (generation_task, k), response, reward in zip(itertools.product(generation_tasks, range(K)), responses, rewards.tolist()):
                program = task.extract_program(response)
                programs.append(program or "")
                with open(os.path.join(self.config.output_dir, f"circle-{l}-{generation_task.chain_idx}-{k}.py"), "w") as f:
                    f.write(programs[-1])
                if math.isfinite(reward):
                    node = Node(id=uuid.uuid4().hex, code=program, parent_ids=list(generation_task.inspiration_ids), gen_id=generation_task.gen_id, chain_idx=generation_task.chain_idx, metrics={"combined_score": reward}, score=reward, status=Status.DONE)
                    self.db.add(node)
                    pending = self.selector.on_child_done(node, [self.db.nodes[parent_id] for parent_id in node.parent_ids])
                else:
                    pending = self.selector.on_generation_failed(generation_task.gen_id)
                if pending is not None:
                    asyncio.run(self.selector.finalize_batch(pending, self.db))
            objective_evaluations = (l + 1) * C * K
            extra = {"best": self.db.best().score}
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards, extra=extra)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=programs)

        wandb.finish()


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
