"""Inspiration selectors over C chains of responses: SimpleTES's RPUCG policy and Top, each chain's best."""

import collections
import math

import torch

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


class Top:
    """C chains, each over its own population of responses rooted at the seed, its inspirations its N best of distinct
    rewards.

    Attributes:
        responses: list (S) of str, every chain's population
        rewards: list (S) of float, finite
        populations: list (C) of list of int, each chain's nodes
    """

    def __init__(self, response, reward, C):
        """
        Args:
            response: str, the seed
            reward: float, finite, the seed's
            C: int, chains
        """
        self.responses, self.rewards = [response], [reward]
        self.populations = [[0] for _ in range(C)]

    def select(self, N):
        """Each chain's inspirations.

        Args:
            N: int, inspirations per chain at most

        Returns:
            list (C) of list of int, each chain's best of distinct rewards, best first
        """
        all_inspirations = []
        for population in self.populations:
            inspirations, rewards = [], set()
            for s in sorted(population, key=self.rewards.__getitem__, reverse=True):
                if len(inspirations) == N:
                    break
                if self.rewards[s] not in rewards:
                    inspirations.append(s)
                    rewards.add(self.rewards[s])
            all_inspirations.append(inspirations)
        return all_inspirations

    def update(self, all_inspirations, all_responses, all_rewards):
        """Adds each chain's responses with a finite reward to its population.

        Args:
            all_inspirations: list (C) of list of int, from select
            all_responses: list (C) of list (K) of str
            all_rewards: list (C) of list (K) of float
        """
        for population, responses, rewards in zip(self.populations, all_responses, all_rewards):
            for response, reward in zip(responses, rewards):
                if math.isfinite(reward):
                    population.append(len(self.responses))
                    self.responses.append(response)
                    self.rewards.append(reward)
