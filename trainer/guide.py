"""Mutates config.task.name's seed program toward a higher reward by sampling config.model.name inside SGLang, each
response pivoted on the in-context distribution of the best responses so far and guided by their NFT weights, no
training.

The buffer holds up to config.G responses with distinct rewards, best first, starting as the task's seed program alone,
taken as a response, so there is no initial generation. Every one of config.L iterations samples config.N new responses
independently, each under log p̂ + Σ_i w_i log p_i over the buffer's G responses. The pivot p̂ is the model under the
task's prompts_with_scores of the top config.M ≤ config.G responses with their rewards; p_i is the model under the
task's prompts of response i alone. Both think at config.model.reasoning_effort when config.model.enable_thinking,
scoring the response after their own prompt. A response's answer is what follows its </think> when thinking, empty when
its thinking never ends, and the whole response otherwise. The weights are NFT's on-policy coefficients of the min-max
normalised rewards r̃_i = (r_i - r_min) / (r_max - r_min): w_i = (r̃_i - r̄) / (G (1 - r̄)), r̄ the mean of r̃, so they
sum to 0, are negative exactly below the mean and give the best 1/G. All rewards tied gives w = 0, each response
sampling from p̂ alone.

The N responses are sampled together by the pipeline's group_generate, each as one group of G + 1 requests, request k
holding guide k's prompt, p̂'s first. Each token of a response is sampled from the softmax of its log p̂ + Σ_i w_i log p_i
at the model's generation config's temperature, with no top-k, top-p or repetition penalty, and given to all of the
group's requests, so its guides advance and stop together; a response fills config.model.max_total_tokens with its
group's longest prompt at most.

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


def guide_weights(rewards):
    """Each buffer program's weight, NFT's on-policy coefficient of its min-max normalised reward.

    Args:
        rewards: list (G) of float, finite

    Returns:
        list (G) of float, w_i = (r̃_i - r̄) / (G (1 - r̄)) summing to 0, with r̃_i = (r_i - r_min) / (r_max - r_min)
        and r̄ the mean of r̃; all 0 when all rewards tie
    """
    rewards = torch.tensor(rewards, dtype=torch.float64)
    rewards_normalised = torch.nan_to_num((rewards - rewards.min()) / (rewards.max() - rewards.min()))
    reward_mean = rewards_normalised.mean()
    return ((rewards_normalised - reward_mean) / (len(rewards) * (1 - reward_mean))).tolist()


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L iterations of guided sampling, evaluation and buffer update from the seed program."""
        task, L, N, G, M = self.task, self.config.L, self.config.N, self.config.G, self.config.M
        buffer = {task.evaluate([task.init_program])[0]: task.init_program}  # reward -> response, best first
        for l in range(L):
            rewards_buffer, responses_buffer = list(buffer), list(buffer.values())
            prompts = [task.prompts_with_scores(responses_buffer[:M], rewards_buffer[:M])] + task.prompts(responses_buffer)
            weights = [1.0] + guide_weights(rewards_buffer)
            responses = self.pipeline.group_generate([prompts] * N, [weights] * N)
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
