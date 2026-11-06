"""Mutates config.task.name's seed program toward a higher reward, alternating agent.py's in-context sampling and
guide.py's guided sampling over config.C chains, no training.

Every one of config.L steps samples config.K responses per chain. An agent step samples as agent.py does, and its prompt
and finite rewards join the chain's PromptsRewards. Step l is a guide step of every chain when l is odd and every
chain's PromptsRewards.get(config.M) holds at least 2 prompts, sampling under Σ_i w_i log p_i, p_i being the model under
its prompt i and w_i guide.py's weights of its aggregated rewards; any other step is an agent step of every chain.

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

from base import BaseTrainer
from guide import weights
from selector import Rpucg

FLAGS = flags.FLAGS


class PromptsRewards:
    """A chain's agent prompts, each with all its finite rewards.

    Attributes:
        data: list (S) of tuple of str and float, each finite reward with its prompt, latest last
    """

    def __init__(self, reward_agg_fn):
        """
        Args:
            reward_agg_fn: str, "mean" or "max", a prompt's aggregated reward over its rewards
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
        """The latest M prompts of distinct aggregated rewards, each tie first reduced to a uniformly picked prompt.

        Args:
            M: int, prompts at most

        Returns:
            list (at most M) of str, the prompts, latest last; list (at most M) of float, their aggregated rewards
        """
        data = pd.DataFrame(self.data, columns=["prompt", "reward"]).reset_index()
        data = data.groupby("prompt").agg(latest=("index", "max"), reward=("reward", self.reward_agg_fn))
        # Shuffled, so that drop_duplicates keeps a uniform pick of each tie.
        data = data.sample(frac=1).drop_duplicates("reward").sort_values("latest").tail(M)
        return data.index.tolist(), data["reward"].tolist()


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L agent or guide steps, evaluation and update from the seed program."""
        task, L, C, K, M, N = self.task, self.config.L, self.config.C, self.config.K, self.config.M, self.config.N
        selector = Rpucg(task.init_program, task.evaluate([task.init_program])[0], C)
        all_prompts_rewards = [PromptsRewards(self.config.reward_agg_fn) for _ in range(C)]
        for l in range(L):
            all_prompts, all_rewards_aggregated = zip(*[prompts_rewards.get(M) for prompts_rewards in all_prompts_rewards])
            guiding = l % 2 == 1 and all(len(prompts) >= 5 for prompts in all_prompts)
            if guiding:
                all_inspirations = [[] for _ in range(C)]
                responses = self.pipeline.group_generate(
                    [prompts for prompts in all_prompts for _ in range(K)],
                    [weights(rewards_aggregated) for rewards_aggregated in all_rewards_aggregated for _ in range(K)],
                )
            else:
                all_inspirations = selector.select(N)
                prompts = [task.prompts_with_scores([selector.responses[s] for s in inspirations], [selector.rewards[s] for s in inspirations]) for inspirations in all_inspirations]
                responses = self.pipeline.generate([prompt for prompt in prompts for _ in range(K)])
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            all_rewards = rewards.view(C, K).tolist()
            selector.update(all_inspirations, [responses[c * K:(c + 1) * K] for c in range(C)], all_rewards)

            if not guiding:
                for c, prompts_rewards in enumerate(all_prompts_rewards):
                    prompts_rewards.set(prompts[c], all_rewards[c])

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/agent-guide.py", "Agent-guide configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
