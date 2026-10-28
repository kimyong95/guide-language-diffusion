"""Runs SimpleTES on config.task's seed program with config.model.name inside vLLM, in synchronous steps.

The task supplies SimpleTES's own generator, for prompts, and its evaluate, running SimpleTES's EvaluatorWorker as its
engine does. The seed is evaluated EngineConfig.init_eval_repeats times, keeping the best metrics.

Every one of config.L steps samples each of config.C chains once. SimpleTES's balance policy picks up to config.N of the
chain's nodes, the generator's prompt holds them with their metrics and reflections and the chain's failure patterns,
and the model answers it config.K times under its generation config, thinking at config.model.reasoning_effort when
config.model.enable_thinking.
Each chain's best new node then gets its reflection, as the policy's reflection_mode would: the model answers
SimpleTES's reflection prompt over that node's prompt, code and metrics, sampled as for generation, and the answer is
kept only when it starts with "Approach". The policy then appends that node to the chain.

A response's code is saved to config.output_dir as circle-{l}-{chain_idx}-{k}.py. A response without an EVOLVE-BLOCK
or a non-finite combined_score is dropped; a code with an evaluator error but a finite combined_score joins the
database at score -inf, its error counting toward the chain's failure patterns. Each step logs every reward and code
to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/simpletes-baseline.py --config config/simpletes-baseline.py
"""

import asyncio
import itertools
import os
import random
import sys
import uuid

import torch
import wandb
from absl import flags
from ml_collections import config_flags
from simpletes.config import EngineConfig
from simpletes.node import Node, NodeDatabase, Status, score_from_metrics
from simpletes.policies.balance import BalancePolicy
from simpletes.policies.base import _MAX_REFLECTION_OUTPUT_CHARS
from simpletes.utils.text import extract_approach_insight, metrics_to_text

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
        init_metrics = max(task.evaluate([task.init_program] * EngineConfig.init_eval_repeats), key=score_from_metrics)
        init_metrics.pop("code")
        self.db.add(Node(id=uuid.uuid4().hex, code=task.init_program, metrics=init_metrics, score=score_from_metrics(init_metrics), status=Status.DONE))
        for l in range(L):
            generation_tasks = []
            for _ in range(C):
                inspirations, failure_patterns, chain_idx = self.selector.select(self.db, N, stats=None, eval_concurrency=0, backpressure_multiplier=0)
                generation_task = task.generator.create_task_with_gen_id(
                    prompt=task.generator.build_prompt(inspirations, failure_patterns),
                    inspiration_ids=[node.id for node in inspirations],
                    k=K,
                    chain_idx=chain_idx,
                    gen_id=task.generator.next_gen_id(),
                )
                self.selector.register_batch(generation_task.gen_id, chain_idx, generation_task.inspiration_ids, K)
                generation_tasks.append(generation_task)
            prompt_tokens = self.pipeline.texts_to_tokens([generation_task.prompt for generation_task in generation_tasks], enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort)
            responses = self.pipeline.generate([tokens for tokens in prompt_tokens for _ in range(K)], enable_thinking=cfg.enable_thinking).texts
            metrics = task.evaluate(responses)
            codes = [metrics_i.pop("code") for metrics_i in metrics]
            rewards = torch.tensor([score_from_metrics(metrics_i) for metrics_i in metrics], dtype=torch.float64)

            pendings = []
            for (generation_task, k), code, metrics_i, reward in zip(itertools.product(generation_tasks, range(K)), codes, metrics, rewards.tolist()):
                with open(os.path.join(self.config.output_dir, f"circle-{l}-{generation_task.chain_idx}-{k}.py"), "w") as f:
                    f.write(code or "")
                node = Node(id=uuid.uuid4().hex, code=code,parent_ids=list(generation_task.inspiration_ids), gen_id=generation_task.gen_id, chain_idx=generation_task.chain_idx, metrics=metrics_i, score=reward, status=Status.DONE, llm_input=generation_task.prompt)
                try:
                    self.db.add(node)
                except ValueError:
                    pending = self.selector.on_generation_failed(generation_task.gen_id)
                else:
                    pending = self.selector.on_child_done(node, [self.db.nodes[parent_id] for parent_id in node.parent_ids])
                if pending is not None:
                    pendings.append(pending)

            best_nodes = [node for node in (self.selector._find_best_node(self.db, pending.children) for pending in pendings) if node is not None]
            if best_nodes:
                reflection_prompts = [self.selector.REFLECTION_TEMPLATE.format(llm_input=node.llm_input, code=node.code, metrics=metrics_to_text(node.metrics)) for node in best_nodes]
                reflection_prompt_tokens = self.pipeline.texts_to_tokens(reflection_prompts, enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort)
                for node, text in zip(best_nodes, self.pipeline.generate(reflection_prompt_tokens, enable_thinking=cfg.enable_thinking).texts):
                    node.reflection = extract_approach_insight(text)[:_MAX_REFLECTION_OUTPUT_CHARS].rstrip()
            for pending in pendings:
                asyncio.run(self.selector.finalize_batch(pending, self.db))

            objective_evaluations = (l + 1) * C * K
            extra = {"best": self.db.best().score}
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards, extra=extra)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=[code or "" for code in codes])

        wandb.finish()


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
