"""Mutates config.task.name's seed program toward a higher reward, alternating agent.py's in-context sampling and
guide.py's guided sampling over config.C chains, no training.

Every one of config.L steps samples config.K responses per chain. An agent step samples as agent.py does, and its prompt
and finite rewards join the chain's PromptsRewards. Step l is a guide step of every chain when l is odd and every
chain's PromptsRewards.get(config.M) holds at least config.M prompts, sampling as guide.py does over those prompts and
their utilities, p_i being the model under prompt i. Any other step is an agent step of every chain.

A guide step's responses join the population by the selector's update with no parents, raising no n, and their rewards
join no PromptsRewards. Evaluation and logging are as in agent.py.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/agent-guide.py --config config/agent-guide.py
"""

import math
import sys

import pandas as pd
import torch
import wandb
from absl import flags
from ml_collections import config_flags

from trainer.base import BaseTrainer
from trainer.selector import Rpucg

FLAGS = flags.FLAGS


class PromptsRewards:
    """A chain's agent prompts, each with all its finite rewards.

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


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L agent or guide steps, evaluation and update from the seed program."""
        task, L, C, K, M, N = self.task, self.config.L, self.config.C, self.config.K, self.config.M, self.config.N
        selector = Rpucg(task.init_program, task.evaluate([task.init_program])[0], C)
        all_prompts_rewards = [PromptsRewards(self.config.reward_agg_fn) for _ in range(C)]
        for l in range(L):
            all_prompts, all_utilities = zip(*[prompts_rewards.get(M) for prompts_rewards in all_prompts_rewards])
            guiding = l % 2 == 1 and all(len(prompts) >= M for prompts in all_prompts)
            if guiding:
                all_in_contexts = [[] for _ in range(C)]
                responses = self.pipeline.group_generate([prompts for prompts in all_prompts for _ in range(K)], [utilities for utilities in all_utilities for _ in range(K)])
            else:
                all_in_contexts = selector.select(N)
                prompts = [task.prompts_with_scores([selector.responses[s] for s in in_contexts], [selector.rewards[s] for s in in_contexts]) for in_contexts in all_in_contexts]
                responses = self.pipeline.generate([prompt for prompt in prompts for _ in range(K)])
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            all_rewards = rewards.view(C, K).tolist()
            selector.update(all_in_contexts, [responses[c * K:(c + 1) * K] for c in range(C)], all_rewards)

            if not guiding:
                for c, prompts_rewards in enumerate(all_prompts_rewards):
                    prompts_rewards.set(prompts[c], all_rewards[c])

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/agent-guide.py", "Agent-guide configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
