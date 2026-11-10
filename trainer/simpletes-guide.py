"""Runs simpletes-baseline.py's SimpleTES, alternating its in-context sampling and guide.py's guided sampling over
config.C chains, no training.

Every one of config.L steps samples config.K responses per chain. A SimpleTES step samples as simpletes-baseline.py
does, and its prompt and finite rewards join the chain's PromptsRewards. Step l is a guide step of every chain when l is
odd and every chain's PromptsRewards.get(config.M) holds at least config.M prompts, sampling as guide.py does over those
prompts and their utilities, p_i being the model under prompt i. Any other step is a SimpleTES step of every chain.

A guide step's responses are a batch of their chain with no parents, raising no visit count, their llm_input the prompt
of the highest utility; their evaluation, reflection and commit to the chain are a SimpleTES step's, and their rewards
join no PromptsRewards. Logging is as in simpletes-baseline.py.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/simpletes-guide.py --config config/simpletes-guide.py
"""

import asyncio
import importlib
import itertools
import math
import sys
import uuid

import torch
import wandb
from absl import flags
from ml_collections import config_flags
from simpletes.config import EngineConfig
from simpletes.node import Node, Status, score_from_metrics
from simpletes.utils.code_extract import extract_code_detailed

SimpletesTrainer = importlib.import_module("trainer.simpletes-baseline").Trainer
PromptsRewards = importlib.import_module("trainer.agent-guide").PromptsRewards

FLAGS = flags.FLAGS


class Trainer(SimpletesTrainer):

    def run(self):
        """Runs config.L SimpleTES or guide steps from the seed program."""
        L, C, K, M, N = self.config.L, self.config.C, self.config.K, self.config.M, self.config.N
        init_metrics = max(self.task.evaluate([self.init_program] * EngineConfig.init_eval_repeats), key=score_from_metrics)
        self.db.add(Node(id=uuid.uuid4().hex, code=self.init_program, metrics=init_metrics, score=score_from_metrics(init_metrics), status=Status.DONE))
        all_prompts_rewards = [PromptsRewards(self.config.reward_agg_fn) for _ in range(C)]
        for l in range(L):
            all_prompts, all_utilities = zip(*[prompts_rewards.get(M) for prompts_rewards in all_prompts_rewards])
            guiding = l % 2 == 1 and all(len(prompts) >= M for prompts in all_prompts)
            selections = []
            for c in range(C):
                if guiding:
                    in_contexts, failure_patterns, chain_idx = [], {}, c
                else:
                    in_contexts, failure_patterns, chain_idx = self.selector.select(self.db, N, stats=None, eval_concurrency=0, backpressure_multiplier=0)
                gen_id = self.generator.next_gen_id()
                self.selector.register_batch(gen_id, chain_idx, [node.id for node in in_contexts], K)
                selections.append((in_contexts, failure_patterns, chain_idx, gen_id))
            all_in_contexts, all_failure_patterns, chain_idxs, gen_ids = zip(*selections)
            if guiding:
                prompts = [prompts[utilities.index(max(utilities))] for prompts, utilities in zip(all_prompts, all_utilities)]
                responses = self.pipeline.group_generate([prompts for prompts in all_prompts for _ in range(K)], [utilities for utilities in all_utilities for _ in range(K)])
            else:
                prompts, responses = self.generate(all_in_contexts, all_failure_patterns)

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

            rewards = torch.tensor([-math.inf if node is None else node.score for node in nodes], dtype=torch.float64)
            if not guiding:
                for c, rewards_c in enumerate(rewards.view(C, K).tolist()):
                    all_prompts_rewards[chain_idxs[c]].set(prompts[c], rewards_c)

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/simpletes-guide.py", "SimpleTES-guide configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
