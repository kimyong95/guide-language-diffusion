"""Selectors over C chains of responses: Rpucg of in-context examples, for SimpleTES's RPUCG, and PromptsRewards of a
chain's guide prompts."""

import collections
import math

import pandas as pd
import torch

PUCT_C = 0.5
GAMMA = 0.8


class Rpucg:
    """SimpleTES's RPUCG policy: C chains over one population of responses of finite reward, each chain's history
    rooted at the seed and holding its best response of every batch with any finite reward.

    A chain's in-context examples are up to N of its history's nodes s by Q(s) + PUCT_C P(s) √(1 + T) / (1 + n(s)),
    picked greedily, each pick excluding its parents and children from the rest. P and Q are the percentile ranks, over
    the population, of the reward r and of V(s) = max(r(s), GAMMA max_children V); T is the chain's batches so far and
    n(s) the times s was among its in-context examples, both counting batches whose rewards are all non-finite.

    Attributes:
        responses: list (S) of str, the population, each child after its parents
        rewards: list (S) of float
        all_parents: list (S) of list of int, each response's parents
        all_children: list (S) of list of int, each response's children
        histories: list (C) of list of int, each chain's nodes, of finite reward
        all_visits: list (C) of collections.Counter, each chain's n
        expansions: list (C) of int, each chain's T
    """

    def __init__(self, response, reward, C):
        """
        Args:
            response: str, the seed
            reward: float, finite, the seed's
            C: int, chains
        """
        self.responses, self.rewards, self.all_parents, self.all_children = [response], [reward], [[]], [[]]
        self.histories = [[0] for _ in range(C)]
        self.all_visits = [collections.Counter() for _ in range(C)]
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
        """Each chain's in-context examples.

        Args:
            N: int, in-context examples per chain at most

        Returns:
            list (C) of list of int, each chain's in-context examples, best first
        """
        values = list(self.rewards)
        for s in reversed(range(len(values))):
            for child in self.all_children[s]:
                values[s] = max(values[s], GAMMA * values[child])
        Q = self.percentile_ranks(torch.tensor(values, dtype=torch.float64)).tolist()
        P = self.percentile_ranks(torch.tensor(self.rewards, dtype=torch.float64)).tolist()

        all_in_contexts = []
        for history, visits, T in zip(self.histories, self.all_visits, self.expansions):
            scores = {s: Q[s] + PUCT_C * P[s] * math.sqrt(1 + T) / (1 + visits[s]) for s in history}
            in_contexts, excluded = [], set()
            for s in sorted(history, key=scores.get, reverse=True):
                if len(in_contexts) == N:
                    break
                if s not in excluded:
                    in_contexts.append(s)
                    excluded.update([s, *self.all_parents[s], *self.all_children[s]])
            all_in_contexts.append(sorted(in_contexts, key=self.rewards.__getitem__, reverse=True))
        return all_in_contexts

    def update(self, all_in_contexts, all_responses, all_rewards):
        """Adds each chain's responses of finite reward to the population as children of its in-context examples,
        appends its best of them to its history and counts the batch, a batch of no finite reward included.

        Args:
            all_in_contexts: list (C) of list of int, from select
            all_responses: list (C) of list (K) of str
            all_rewards: list (C) of list (K) of float
        """
        for c, (in_contexts, responses, rewards) in enumerate(zip(all_in_contexts, all_responses, all_rewards)):
            children = []
            for response, reward in zip(responses, rewards):
                if not math.isfinite(reward):
                    continue
                children.append(len(self.responses))
                self.responses.append(response)
                self.rewards.append(reward)
                self.all_parents.append(in_contexts)
                self.all_children.append([])
                for parent in in_contexts:
                    self.all_children[parent].append(children[-1])
            if children:
                self.histories[c].append(max(children, key=self.rewards.__getitem__))
            self.all_visits[c].update(in_contexts)
            self.expansions[c] += 1


class PromptsRewards:
    """A chain's prompts, each with all its finite rewards.

    Attributes:
        data: list (S) of tuple of str and float, each finite reward with its prompt, latest last
    """

    def __init__(self, reward_agg_fn):
        """
        Args:
            reward_agg_fn: str, "mean" or "max", a prompt's utility over its rewards
        """
        self.reward_agg_fn = reward_agg_fn
        self.data = []

    def set(self, prompt, rewards):
        """Adds a prompt's finite rewards, any of them making it the latest prompt.

        Args:
            prompt: str
            rewards: list of float
        """
        self.data += [(prompt, reward) for reward in rewards if math.isfinite(reward)]

    def get(self, M):
        """The latest M prompts of distinct utilities, each tie first reduced to a uniformly picked prompt.

        Args:
            M: int, prompts at most

        Returns:
            list (at most M) of str, the prompts, latest last; list (at most M) of float, their utilities
        """
        data = pd.DataFrame(self.data, columns=["prompt", "reward"]).reset_index()
        data = data.groupby("prompt").agg(latest=("index", "max"), utility=("reward", self.reward_agg_fn))
        # Shuffled, so that drop_duplicates keeps a uniform pick of each tie.
        data = data.sample(frac=1).drop_duplicates("utility").sort_values("latest").tail(M)
        return data.index.tolist(), data["utility"].tolist()
