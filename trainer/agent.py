"""Mutates config.task.name's seed program toward a higher reward by in-context sampling of config.model.name inside
SGLang, its in-context examples picked by SimpleTES's RPUCG policy over config.C chains, no guidance and no training.

The population holds every response with a finite reward, starting as the task's seed program alone, taken as a
response, the root of every chain. Every one of config.L steps samples each chain once: RPUCG picks up to config.N of the
chain's nodes, and config.K new responses are sampled independently under the task's prompts_with_scores of them with
their rewards, best first, thinking at config.model.reasoning_effort when config.model.enable_thinking; all C × K
responses come from one generate call. A response's answer is what follows its </think> when thinking, empty when its
thinking never ends, and the whole response otherwise.

Each response's reward r is the task's evaluate, config.task.concurrency at a time. A response with a finite reward joins
the population as a child of its prompt's nodes, and the chain's best of them joins its history. Each step logs every
reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/agent.py --config config/agent.py
"""

import collections
import math
import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from base import BaseTrainer

FLAGS = flags.FLAGS

PUCT_C = 0.5
GAMMA = 0.8


class RPUCG:
    """SimpleTES's RPUCG policy: C chains over one population of responses, each chain's history rooted at the seed and
    holding its best response of every batch.

    A chain's inspirations are up to N of its history's nodes s by Q(s) + PUCT_C P(s) √(1 + T) / (1 + n(s)), picked
    greedily, each pick excluding its parents and children from the rest. P and Q are the percentile ranks, over the
    population, of the reward r and of V(s) = max(r(s), GAMMA max_children V); T is the chain's batches so far and n(s)
    the times s was among its inspirations.

    Attributes:
        responses: list (S) of str, the population, each child after its parents
        rewards: list (S) of float, finite
        parents: list (S) of list of int
        children: list (S) of list of int
        histories: list (C) of list of int, each chain's nodes
        visits: list (C) of collections.Counter, each chain's n
        expansions: list (C) of int, each chain's T
    """

    def __init__(self, response, reward, C):
        """
        Args:
            response: str, the seed
            reward: float, finite, the seed's
            C: int, chains
        """
        self.responses, self.rewards, self.parents, self.children = [response], [reward], [[]], [[]]
        self.histories = [[0] for _ in range(C)]
        self.visits = [collections.Counter() for _ in range(C)]
        self.expansions = [0] * C

    @staticmethod
    def percentile_ranks(values):
        """Each value's percentile rank, the fraction of the values strictly below it.

        Args:
            values: (S) float64

        Returns:
            (S) float64
        """
        return torch.searchsorted(values.sort().values, values) / len(values)

    def select(self, N):
        """Each chain's inspirations.

        Args:
            N: int, inspirations per chain at most

        Returns:
            list (C) of list of int, each chain's inspirations, best first
        """
        values = list(self.rewards)
        for s in reversed(range(len(values))):
            for child in self.children[s]:
                values[s] = max(values[s], GAMMA * values[child])
        Q = self.percentile_ranks(torch.tensor(values, dtype=torch.float64)).tolist()
        P = self.percentile_ranks(torch.tensor(self.rewards, dtype=torch.float64)).tolist()

        all_inspirations = []
        for history, visits, T in zip(self.histories, self.visits, self.expansions):
            scores = {s: Q[s] + PUCT_C * P[s] * math.sqrt(1 + T) / (1 + visits[s]) for s in history}
            inspirations, excluded = [], set()
            for s in sorted(history, key=scores.get, reverse=True):
                if len(inspirations) == N:
                    break
                if s not in excluded:
                    inspirations.append(s)
                    excluded.update([s, *self.parents[s], *self.children[s]])
            all_inspirations.append(sorted(inspirations, key=self.rewards.__getitem__, reverse=True))
        return all_inspirations

    def update(self, all_inspirations, all_responses, all_rewards):
        """Adds each chain's responses with a finite reward to the population as children of its inspirations, appends
        its best of them to its history and counts the batch.

        Args:
            all_inspirations: list (C) of list of int, from select
            all_responses: list (C) of list (K) of str
            all_rewards: list (C) of list (K) of float
        """
        for c, (inspirations, responses, rewards) in enumerate(zip(all_inspirations, all_responses, all_rewards)):
            children = []
            for response, reward in zip(responses, rewards):
                if math.isfinite(reward):
                    children.append(len(self.responses))
                    self.responses.append(response)
                    self.rewards.append(reward)
                    self.parents.append(inspirations)
                    self.children.append([])
                    for parent in inspirations:
                        self.children[parent].append(children[-1])
            if children:
                self.histories[c].append(max(children, key=self.rewards.__getitem__))
            self.visits[c].update(inspirations)
            self.expansions[c] += 1


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of RPUCG selection, in-context sampling, evaluation and update from the seed program."""
        task, L, C, K, N = self.task, self.config.L, self.config.C, self.config.K, self.config.N
        rpucg = RPUCG(task.init_program, task.evaluate([task.init_program])[0], C)
        for l in range(L):
            all_inspirations = rpucg.select(N)
            prompts = [task.prompts_with_scores([rpucg.responses[s] for s in inspirations], [rpucg.rewards[s] for s in inspirations]) for inspirations in all_inspirations]
            responses = self.pipeline.generate([prompt for prompt in prompts for _ in range(K)])
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            rpucg.update(all_inspirations, [responses[c * K:(c + 1) * K] for c in range(C)], rewards.view(C, K).tolist())

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/agent.py", "Agent configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
