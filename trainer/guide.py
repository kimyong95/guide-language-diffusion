"""Mutates config.task.name's seed program toward a higher reward by sampling config.model.name inside SGLang, each
response guided by the best responses so far under config.mode's weights, no training.

The buffer holds up to config.G responses with distinct rewards, so no two tie, best first, starting as the task's seed program alone,
taken as a response, so there is no initial generation. Every one of config.L iterations samples config.N new responses
independently, each under a weighted sum of log-probabilities over the buffer's G responses, p_i being the model under
the task's prompts of response i alone. Every guide thinks at config.model.reasoning_effort when
config.model.enable_thinking, scoring the response after its own prompt. A response's answer is what follows its
</think> when thinking, empty when its thinking never ends, and the whole response otherwise. With the min-max
normalised rewards r̃_i = (r_i - r_min) / (r_max - r_min), the modes sample:

- "A": response n under log p_i + Σ_j w_ij log p_j, its anchor i = n mod G taking the buffer round-robin. With
  d_ij = r̃_j - r̃_i and D⁺_i, D⁻_i the sums of i's positive and absolute negative gaps, w_ij = d_ij / D⁺_i for
  d_ij > 0 and d_ij / D⁻_i for d_ij < 0: the better responses' weights sum to 1, the worse ones' to -1. The best
  anchor, with no better response, gets w_ii = 1 in their place and the worst w_ii = -1, so a lone response, as the
  seed alone, samples from itself alone.
- "B": every response under (1/G) Σ_i log p_i + Σ_i w_i log p_i, w_i = (r̃_i - r̄) / (G r̄ (1 - r̄)) with r̄ the
  mean of r̃, so they sum to 0 and are negative exactly below the mean. A lone response gets w = 0, each response
  sampling from it alone.
- "C": every response under Σ_i w_i log p_i, w_i = (r_i - r_min) / Σ_j (r_j - r_min), so they sum to 1 and the worst
  gets 0. A lone response gets w = 1.

The N responses are sampled together by the pipeline's group_generate, each as one group of requests, one per guide.
Each token of a response is sampled from the softmax of its weighted sum at the model's generation config's
temperature, with no top-k, top-p or repetition penalty, and given to all of the group's requests, so its guides
advance and stop together; a response fills config.model.max_total_tokens with its group's longest prompt at most.

Each response's reward r is the task's evaluate, config.task.concurrency at a time. A response joins the buffer unless
its reward is -inf, as without an EVOLVE-BLOCK or when the evaluator rejects its code, or one the buffer already holds;
the buffer then keeps its G highest-reward responses. Each iteration logs every reward and response to wandb, -inf and
"" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/guide.py --config config/guide.py
"""

import math
import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")


def normalise_rewards(rewards):
    """The rewards min-max normalised, r̃_i = (r_i - r_min) / (r_max - r_min); 0 for a lone reward.

    Args:
        rewards: list (G) of float, finite, distinct

    Returns:
        (G) float64
    """
    rewards = torch.tensor(rewards, dtype=torch.float64)
    return torch.nan_to_num((rewards - rewards.min()) / (rewards.max() - rewards.min()))


def guide_weights_a(rewards):
    """Each anchor's weights over the buffer responses, its normalised reward gaps to them normalised per side.

    Args:
        rewards: list (G) of float, finite, distinct, descending

    Returns:
        list (G) of list (G) of float, w[i][j] for anchor i and guide j: d_ij / D⁺_i for d_ij > 0 and d_ij / D⁻_i for
        d_ij < 0, with d_ij = r̃_j - r̃_i and D⁺_i, D⁻_i the sums of i's positive and absolute negative gaps;
        w[i][i] = 1[i = 0] - 1[i = G - 1], standing in for the best's empty positive side and the worst's empty
        negative one; each anchor's summing to 0
    """
    rewards_normalised = normalise_rewards(rewards)
    gaps = rewards_normalised[None, :] - rewards_normalised[:, None]
    gaps_positive, gaps_negative = gaps.clamp(min=0), gaps.clamp(max=0)
    weights = torch.nan_to_num(gaps_positive / gaps_positive.sum(1, keepdim=True)) + torch.nan_to_num(gaps_negative / -gaps_negative.sum(1, keepdim=True))
    weights[0, 0] += 1
    weights[-1, -1] -= 1
    return weights.tolist()


def guide_weights_b(rewards):
    """Each buffer response's weight, its normalised reward's deviation from their mean.

    Args:
        rewards: list (G) of float, finite, distinct

    Returns:
        list (G) of float, w_i = (r̃_i - r̄) / (G r̄ (1 - r̄)) summing to 0, with r̄ the mean of r̃; 0 for a lone reward
    """
    rewards_normalised = normalise_rewards(rewards)
    reward_mean = rewards_normalised.mean()
    return torch.nan_to_num((rewards_normalised - reward_mean) / (len(rewards) * reward_mean * (1 - reward_mean))).tolist()


def guide_weights_c(rewards):
    """Each buffer response's weight, its reward's share above the minimum.

    Args:
        rewards: list (G) of float, finite, distinct

    Returns:
        list (G) of float, w_i = (r_i - r_min) / Σ_j (r_j - r_min) summing to 1; 1 for a lone reward
    """
    rewards_normalised = normalise_rewards(rewards)
    return torch.nan_to_num(rewards_normalised / rewards_normalised.sum(), nan=1.0).tolist()


class Trainer(BaseTrainer):

    def guides(self, responses_buffer, rewards_buffer):
        """Each of the config.N responses' guide prompts and weights under config.mode.

        Args:
            responses_buffer: list (G) of str, best first
            rewards_buffer: list (G) of float, finite, distinct, descending

        Returns:
            all_prompts: list (N) of list (G) of str, each response's guide prompts
            all_weights: list (N) of list (G) of float, each summing to 1
        """
        task, N, G = self.task, self.config.N, len(responses_buffer)
        prompts = task.prompts(responses_buffer)
        if self.config.mode == "A":
            weights = guide_weights_a(rewards_buffer)
            anchors = [n % G for n in range(N)]
            return [prompts] * N, [[float(j == i) + weight for j, weight in enumerate(weights[i])] for i in anchors]
        if self.config.mode == "B":
            return [prompts] * N, [[1 / G + weight for weight in guide_weights_b(rewards_buffer)]] * N
        if self.config.mode == "C":
            return [prompts] * N, [guide_weights_c(rewards_buffer)] * N
        raise ValueError(f"unknown config.mode {self.config.mode!r}")

    def run(self):
        """Runs config.L iterations of guided sampling, evaluation and buffer update from the seed program."""
        task, L, N, G = self.task, self.config.L, self.config.N, self.config.G
        buffer = {task.evaluate([task.init_program])[0]: task.init_program}  # reward -> response, best first
        for l in range(L):
            rewards_buffer, responses_buffer = list(buffer), list(buffer.values())
            responses = self.pipeline.group_generate(*self.guides(responses_buffer, rewards_buffer))
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            for response, reward in zip(responses, rewards.tolist()):
                if math.isfinite(reward):
                    buffer.setdefault(reward, response)
            buffer = dict(sorted(buffer.items(), reverse=True)[:G])

            objective_evaluations = (l + 1) * N
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
