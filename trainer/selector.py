"""Inspiration selectors over C chains of responses: SimpleTES's RPUCG policy and Top, each chain's best."""

import collections
import heapq
import math

import torch

PUCT_C = 0.5
GAMMA = 0.8


class RPUCG:
    """SimpleTES's RPUCG policy: C chains over one population of responses, each chain's history rooted at the seed and
    holding its best response of every batch whose reward is finite.

    A chain's inspirations are up to N of its history's nodes s by Q(s) + PUCT_C P(s) √(1 + T) / (1 + n(s)), picked
    greedily, each pick excluding its parents and children from the rest. P and Q are the percentile ranks, over the
    population, non-finite rewards included, of the reward r and of V(s) = max(r(s), GAMMA max_children V); T is the
    chain's batches so far and n(s) the times s was among its inspirations.

    Attributes:
        responses: list (S) of str, the population, each child after its parents
        rewards: list (S) of float
        parents: list (S) of list of int
        children: list (S) of list of int
        histories: list (C) of list of int, each chain's nodes, of finite reward
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
        """Adds each chain's responses to the population as children of its inspirations, appends its best of them to
        its history when that reward is finite and counts the batch.

        Args:
            all_inspirations: list (C) of list of int, from select
            all_responses: list (C) of list (K) of str
            all_rewards: list (C) of list (K) of float
        """
        for c, (inspirations, responses, rewards) in enumerate(zip(all_inspirations, all_responses, all_rewards)):
            children = []
            for response, reward in zip(responses, rewards):
                children.append(len(self.responses))
                self.responses.append(response)
                self.rewards.append(reward)
                self.parents.append(inspirations)
                self.children.append([])
                for parent in inspirations:
                    self.children[parent].append(children[-1])
            best = max(children, key=self.rewards.__getitem__)
            if math.isfinite(self.rewards[best]):
                self.histories[c].append(best)
            self.visits[c].update(inspirations)
            self.expansions[c] += 1


class Top:
    """C chains, each over its own population of responses rooted at the seed.

    A chain's guides are its G combinations of N inspirations of distinct rewards of the highest average reward, its G
    best when N = 1.

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

    def select(self, G, N):
        """Each chain's guides, its combinations of the highest average reward.

        A combination holds N of the chain's nodes of distinct rewards, so N = 1 gives its G best. Best-first search from
        the N best: a successor moves one inspiration to the next reward, never raising the sum.

        Args:
            G: int, guides per chain at most
            N: int, inspirations per guide at most

        Returns:
            list (C) of list of list of int, each chain's guides, best first, each guide's inspirations, best first
        """
        all_guides = []
        for population in self.populations:
            nodes = sorted({self.rewards[s]: s for s in reversed(population)}.values(), key=self.rewards.__getitem__, reverse=True)
            rewards = [self.rewards[s] for s in nodes]
            first = tuple(range(min(N, len(nodes))))
            heap, seen, guides = [(-sum(rewards[i] for i in first), first)], {first}, []
            while heap and len(guides) < G:
                _, combination = heapq.heappop(heap)
                guides.append([nodes[i] for i in combination])
                for j, i in enumerate(combination):
                    successor = (*combination[:j], i + 1, *combination[j + 1:])
                    if i + 1 < len(nodes) and i + 1 not in combination and successor not in seen:
                        seen.add(successor)
                        heapq.heappush(heap, (-sum(rewards[k] for k in successor), successor))
            all_guides.append(guides)
        return all_guides

    def update(self, all_guides, all_responses, all_rewards):
        """Adds each chain's responses with a finite reward to its population.

        Args:
            all_guides: list (C), from select
            all_responses: list (C) of list (K) of str
            all_rewards: list (C) of list (K) of float
        """
        for population, responses, rewards in zip(self.populations, all_responses, all_rewards):
            for response, reward in zip(responses, rewards):
                if math.isfinite(reward):
                    population.append(len(self.responses))
                    self.responses.append(response)
                    self.rewards.append(reward)
