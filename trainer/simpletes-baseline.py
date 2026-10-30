"""Runs SimpleTES on config.task.name's seed program with config.model.name inside vLLM, in synchronous steps.

config.task.name names a tasks.py SimpleTES task, e.g. "circle-packing-26", whose seed program, SimpleTES generator, for
prompts, and EvaluatorWorker, built as SimpleTES's engine builds them, the trainer takes. A response's code is the seed
with its EVOLVE-BLOCK replaced by the response's, merged by SimpleTES's extract_code_detailed. The task's evaluate, its
evaluate_one replaced by the trainer's to return metrics, runs each code's EvaluatorWorker in a fresh subprocess,
config.task.concurrency at a time, killed after EngineConfig's eval_timeout, and its error is truncated to
DEFAULT_METRICS_ERROR_MAX_CHARS. The seed is evaluated EngineConfig.init_eval_repeats times, keeping the best metrics.

Every one of config.L steps samples each of config.C chains once. SimpleTES's RPUCG policy, at EngineConfig's puct_c and
rpucg_gamma as SimpleTES's engine passes them, picks up to config.N of the chain's nodes, the generator's prompt holds
them with their metrics and reflections and the chain's failure patterns, and the model answers it config.K times under
its generation config, thinking at config.model.reasoning_effort when config.model.enable_thinking.
Each chain's best new node then gets its reflection, as the policy's reflection_mode would: the model answers the
policy's reflection prompt over that node's prompt, code and metrics, sampled as for generation, and the answer is
kept only when it starts with "Approach". The policy then appends that node to the chain.

A response without an EVOLVE-BLOCK is a failed generation, never evaluated, and a code with a non-finite combined_score
is dropped; a code with an evaluator error but a finite combined_score joins the database at score -inf, its error
counting toward the chain's failure patterns. Each step logs every reward and code to wandb, a failed generation's as
-inf and "".

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/simpletes-baseline.py --config config/simpletes-baseline.py
"""

import asyncio
import itertools
import math
import random
import sys
import uuid

import torch
import wandb
from absl import flags
from ml_collections import config_flags
from simpletes.config import EngineConfig
from simpletes.node import Node, NodeDatabase, Status, score_from_metrics
from simpletes.policies.base import _MAX_REFLECTION_OUTPUT_CHARS
from simpletes.policies.rpucg import RpucgPolicy
from simpletes.utils.code_extract import extract_code_detailed
from simpletes.utils.text import DEFAULT_METRICS_ERROR_MAX_CHARS, extract_approach_insight, metrics_to_text, truncate_error_in_metrics

from base import BaseTrainer

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)
        random.seed(config.seed)
        self.db = NodeDatabase()
        self.selector = RpucgPolicy(
            num_chains=config.C,
            max_generations=config.L * config.C * config.K,
            k=config.K,
            c=EngineConfig.puct_c,
            gamma=EngineConfig.rpucg_gamma,
        )

    def setup_task(self):
        """Builds config.task.name's tasks.py task with evaluate_one replaced, taking its seed program, EVOLVE-BLOCK
        context, generator and EvaluatorWorker."""
        super().setup_task()
        self.task.evaluate_one = self.evaluate_one
        self.init_program, self.evolve_context = self.task.init_program, self.task.evolve_context
        self.generator, self.evaluator = self.task.generator, self.task.evaluator

    async def evaluate_one(self, code, semaphore):
        """SimpleTES's engine's _evaluate_code, holding the semaphore during the evaluation; the task's evaluate_one,
        so that its evaluate returns each code's metrics."""
        async with semaphore:
            try:
                metrics = (await self.evaluator.evaluate(code)).metrics
            except Exception as exception:
                metrics = {"error": str(exception), "combined_score": -math.inf}
        return truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)

    def generate(self, all_inspirations, all_failure_patterns):
        """Samples config.K responses per chain by in-context learning: the generator's prompt holds all of the
        chain's inspirations with their metrics and reflections and the chain's failure patterns.

        Args:
            all_inspirations: list (C) of list of Node, each chain's inspirations from the selector
            all_failure_patterns: list (C) of dict of str to float, each chain's failure patterns from the selector

        Returns:
            prompts: list (C) of str, each chain's prompt, its children's llm_input
            responses: list (C × K) of str, chain c's response k at c × K + k, each its answer: after its </think> when
                thinking, empty when its thinking never ends; the whole response otherwise
        """
        cfg, K = self.config.model, self.config.K
        prompts = [self.generator.build_prompt(inspirations, failure_patterns) for inspirations, failure_patterns in zip(all_inspirations, all_failure_patterns)]
        prompt_tokens = self.pipeline.texts_to_tokens(prompts, enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort)
        responses = self.pipeline.generate([tokens for tokens in prompt_tokens for _ in range(K)], enable_thinking=cfg.enable_thinking).texts
        return prompts, responses

    def reflect(self, nodes):
        """Sets each node's reflection from the model's answer to SimpleTES's reflection prompt over its llm_input, code
        and metrics, sampled as for generation."""
        cfg = self.config.model
        reflection_prompts = [self.selector.REFLECTION_TEMPLATE.format(llm_input=node.llm_input, code=node.code, metrics=metrics_to_text(node.metrics)) for node in nodes]
        reflection_prompt_tokens = self.pipeline.texts_to_tokens(reflection_prompts, enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort)
        for node, text in zip(nodes, self.pipeline.generate(reflection_prompt_tokens, enable_thinking=cfg.enable_thinking).texts):
            node.reflection = extract_approach_insight(text)[:_MAX_REFLECTION_OUTPUT_CHARS].rstrip()

    def run(self):
        """Runs config.L steps of SimpleTES from the seed program."""
        L, C, K, N = self.config.L, self.config.C, self.config.K, self.config.N
        init_metrics = max(self.task.evaluate([self.init_program] * EngineConfig.init_eval_repeats), key=score_from_metrics)
        self.db.add(Node(id=uuid.uuid4().hex, code=self.init_program, metrics=init_metrics, score=score_from_metrics(init_metrics), status=Status.DONE))
        for l in range(L):
            selections = []
            for _ in range(C):
                inspirations, failure_patterns, chain_idx = self.selector.select(self.db, N, stats=None, eval_concurrency=0, backpressure_multiplier=0)
                gen_id = self.generator.next_gen_id()
                self.selector.register_batch(gen_id, chain_idx, [node.id for node in inspirations], K)
                selections.append((inspirations, failure_patterns, chain_idx, gen_id))
            all_inspirations, all_failure_patterns, chain_idxs, gen_ids = zip(*selections)
            prompts, responses = self.generate(all_inspirations, all_failure_patterns)

            nodes, pendings = [], []
            for (c, _), response in zip(itertools.product(range(C), range(K)), responses):
                code, _ = extract_code_detailed(response, self.evolve_context)
                if code is None:
                    pendings.append(self.selector.on_generation_failed(gen_ids[c]))
                else:
                    nodes.append(Node(id=uuid.uuid4().hex, code=code, parent_ids=[node.id for node in all_inspirations[c]], gen_id=gen_ids[c], chain_idx=chain_idxs[c], status=Status.EVAL_PENDING, llm_input=prompts[c]))
            for node, metrics in zip(nodes, self.task.evaluate([node.code for node in nodes])):
                node.metrics, node.score, node.status = metrics, score_from_metrics(metrics), Status.DONE
                try:
                    self.db.add(node)
                except ValueError:
                    pendings.append(self.selector.on_generation_failed(node.gen_id))
                else:
                    pendings.append(self.selector.on_child_done(node, [self.db.nodes[parent_id] for parent_id in node.parent_ids]))
            pendings = [pending for pending in pendings if pending is not None]

            best_nodes = [node for node in (self.selector._find_best_node(self.db, pending.children) for pending in pendings) if node is not None]
            if best_nodes:
                self.reflect(best_nodes)
            for pending in pendings:
                asyncio.run(self.selector.finalize_batch(pending, self.db))

            num_failed = len(responses) - len(nodes)
            rewards = torch.tensor([node.score for node in nodes] + [-math.inf] * num_failed, dtype=torch.float64)
            objective_evaluations = (l + 1) * C * K
            extra = {"best": self.db.best().score}
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards, extra=extra)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=[node.code for node in nodes] + [""] * num_failed)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/simpletes-baseline.py", "SimpleTES configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
