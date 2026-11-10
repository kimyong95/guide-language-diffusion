"""Runs SimpleTES on config.task.name's seed program with config.model.name inside SGLang, in synchronous steps.

config.task.name names a tasks.py SimpleTES task, e.g. "circle-packing-26", whose seed program, SimpleTES generator, for
prompts, and EvaluatorWorker, built as SimpleTES's engine builds them, the trainer takes. A response's code is the seed
with its EVOLVE-BLOCK replaced by the response's, merged by SimpleTES's extract_code_detailed. The task's evaluate, its
evaluate_one_async replaced by the trainer's to return metrics, runs each code's EvaluatorWorker in a fresh subprocess,
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
counting toward the chain's failure patterns. Each step logs every response with its reward to wandb, a failed
generation's reward as -inf.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/simpletes-baseline.py --config config/simpletes-baseline.py
"""

import asyncio
import itertools
import math
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

from trainer.base import BaseTrainer

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)
        self.db = NodeDatabase()
        self.selector = RpucgPolicy(
            num_chains=config.C,
            max_generations=config.L * config.C * config.K,
            k=config.K,
            c=EngineConfig.puct_c,
            gamma=EngineConfig.rpucg_gamma,
        )

    def setup_task(self):
        """Builds config.task.name's tasks.py task with evaluate_one_async replaced, taking its seed program, EVOLVE-BLOCK
        context, generator and EvaluatorWorker."""
        super().setup_task()
        self.task.evaluate_one_async = self.evaluate_one_async
        self.init_program, self.evolve_context = self.task.init_program, self.task.evolve_context
        self.generator, self.evaluator = self.task.generator, self.task.evaluator

    async def evaluate_one_async(self, code, semaphore):
        """SimpleTES's engine's _evaluate_code, holding the semaphore during the evaluation; the task's evaluate_one_async,
        so that its evaluate returns each code's metrics."""
        async with semaphore:
            try:
                metrics = (await self.evaluator.evaluate(code)).metrics
            except Exception as exception:
                metrics = {"error": str(exception), "combined_score": -math.inf}
        return truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)

    def generate(self, all_in_contexts, all_failure_patterns):
        """Samples config.K responses per chain by in-context learning: the generator's prompt holds all of the
        chain's in-context examples with their metrics and reflections and the chain's failure patterns.

        Args:
            all_in_contexts: list (C) of list of Node, each chain's in-context examples from the selector
            all_failure_patterns: list (C) of dict of str to float, each chain's failure patterns from the selector

        Returns:
            prompts: list (C) of str, each chain's prompt, its children's llm_input
            responses: list (C × K) of str, chain c's response k at c × K + k, each its answer: after its </think> when
                thinking, empty when its thinking never ends; the whole response otherwise
        """
        K = self.config.K
        prompts = [self.generator.build_prompt(in_contexts, failure_patterns) for in_contexts, failure_patterns in zip(all_in_contexts, all_failure_patterns)]
        responses = self.pipeline.generate([prompt for prompt in prompts for _ in range(K)])
        return prompts, responses

    def reflect(self, nodes):
        """Sets each node's reflection from the model's answer to SimpleTES's reflection prompt over its llm_input, code
        and metrics, sampled as for generation."""
        reflection_prompts = [self.selector.REFLECTION_TEMPLATE.format(llm_input=node.llm_input, code=node.code, metrics=metrics_to_text(node.metrics)) for node in nodes]
        for node, text in zip(nodes, self.pipeline.generate(reflection_prompts)):
            node.reflection = extract_approach_insight(text)[:_MAX_REFLECTION_OUTPUT_CHARS].rstrip()

    def add_init_program(self):
        """Adds the seed program to the database at the best of its EngineConfig.init_eval_repeats metrics."""
        init_metrics = max(self.task.evaluate([self.init_program] * EngineConfig.init_eval_repeats), key=score_from_metrics)
        self.db.add(Node(id=uuid.uuid4().hex, code=self.init_program, metrics=init_metrics, score=score_from_metrics(init_metrics), status=Status.DONE))

    def commit(self, all_in_contexts, chain_idxs, gen_ids, prompts, responses):
        """Evaluates each response's code as a node of its chain's batch, reflects on each batch's best node and commits
        the batches to their chains.

        Args:
            all_in_contexts: list (C) of list of Node, each chain's in-context examples, its nodes' parents
            chain_idxs: list (C) of int, each chain's index
            gen_ids: list (C) of int, each chain's batch, registered with the selector
            prompts: list (C) of str, each chain's prompt, its nodes' llm_input
            responses: list (C × K) of str, chain c's response k at c × K + k

        Returns:
            (C × K) float64, each response's reward: its node's score, -inf without an EVOLVE-BLOCK
        """
        C, K = self.config.C, self.config.K
        nodes, pendings = [], []  # nodes: each response's node, None without an EVOLVE-BLOCK
        for (c, _), response in zip(itertools.product(range(C), range(K)), responses):
            code, _ = extract_code_detailed(response, self.evolve_context)
            if code is None:
                pendings.append(self.selector.on_generation_failed(gen_ids[c]))
                nodes.append(None)
            else:
                nodes.append(Node(id=uuid.uuid4().hex, code=code, parent_ids=[node.id for node in all_in_contexts[c]], gen_id=gen_ids[c], chain_idx=chain_idxs[c], status=Status.EVAL_PENDING, llm_input=prompts[c]))
        nodes_evaluated = [node for node in nodes if node is not None]
        for node, metrics in zip(nodes_evaluated, self.task.evaluate([node.code for node in nodes_evaluated])):
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

        return torch.tensor([-math.inf if node is None else node.score for node in nodes], dtype=torch.float64)

    def run(self):
        """Runs config.L steps of SimpleTES from the seed program."""
        L, C, K, N = self.config.L, self.config.C, self.config.K, self.config.N
        self.add_init_program()
        for l in range(L):
            selections = []
            for _ in range(C):
                in_contexts, failure_patterns, chain_idx = self.selector.select(self.db, N, stats=None, eval_concurrency=0, backpressure_multiplier=0)
                gen_id = self.generator.next_gen_id()
                self.selector.register_batch(gen_id, chain_idx, [node.id for node in in_contexts], K)
                selections.append((in_contexts, failure_patterns, chain_idx, gen_id))
            all_in_contexts, all_failure_patterns, chain_idxs, gen_ids = zip(*selections)
            prompts, responses = self.generate(all_in_contexts, all_failure_patterns)
            rewards = self.commit(all_in_contexts, chain_idxs, gen_ids, prompts, responses)

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/simpletes-baseline.py", "SimpleTES configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
