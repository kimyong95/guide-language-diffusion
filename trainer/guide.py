"""Mutates config.task.name's seed program toward a higher reward by guided sampling of config.model.name inside SGLang,
its guides picked by selector.py's Top over config.C chains, no training.

Every one of config.L steps samples each chain once: Top picks up to N = 3 of the chain's population as its
inspirations, its best of distinct rewards, best first, and config.K new responses are sampled independently under a weighted sum of log-probabilities over them,
p_i being the model under the task's prompts of inspiration i alone. Every guide thinks at config.model.reasoning_effort
when config.model.enable_thinking, scoring the response after its own prompt. A response's answer is what follows its
</think> when thinking, empty when its thinking never ends, and the whole response otherwise. Every response is
sampled under Σ_i w_i log p_i, the weights maximising Σ_i w_i logit r̄_i with Σ_i w_i = 1 and the mean w_i² at most
BOUND, r̄_i being the inspirations' rewards min-max normalised over themselves; uniform when they are all equal.

The C × K responses are sampled together by the pipeline's group_generate, each as one group of requests, one per guide.
Each token of a response is sampled from the softmax of its weighted sum at the model's generation config's temperature,
with no top-k, top-p or repetition penalty, and given to all of the group's requests, so its guides advance and stop
together; a response fills config.model.max_total_tokens with its group's longest prompt at most.

Each response's reward r is the task's evaluate, config.task.concurrency at a time, and each chain's responses with a finite
reward join its population.
Each step logs every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/guide.py --config config/guide.py
"""

import math
import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from base import BaseTrainer
from selector import Top

FLAGS = flags.FLAGS

N = 3
BOUND = 1.0


def logit(r):
    return math.log(r / (1 - r))


def normalised(rewards):
    """The rewards min-max normalised over themselves into [1/(2K), 1 - 1/(2K)], the extremes' rank positions.

    Args:
        rewards: list (K) of float, not all equal

    Returns:
        list (K) of float in (0, 1)
    """
    low, high = min(rewards), max(rewards)
    return [(0.5 + (len(rewards) - 1) * (r - low) / (high - low)) / len(rewards) for r in rewards]


def rms_weights(rewards):
    """The weights maximising Σ_i w_i logit r̄_i with Σ_i w_i = 1 and the mean w_i² at most BOUND, r̄_i the normalised
    rewards; uniform when the rewards are all equal.

    Args:
        rewards: list (K) of float, finite, each guide's inspiration's reward

    Returns:
        list (K) of float, 1/K plus each guide's z-score of logit r̄_i, scaled by √(K² BOUND - 1) / K
    """
    if min(rewards) == max(rewards):
        return [1 / len(rewards)] * len(rewards)
    ell = [logit(r) for r in normalised(rewards)]
    mean = sum(ell) / len(ell)
    sd = math.sqrt(sum((l - mean) ** 2 for l in ell) / len(ell))
    return [1 / len(ell) + math.sqrt(len(ell) ** 2 * BOUND - 1) / len(ell) * (l - mean) / sd for l in ell]


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of Top selection, guided sampling, evaluation and update from the seed program."""
        task, L, C, K = self.task, self.config.L, self.config.C, self.config.K
        top = Top(task.init_program, task.evaluate([task.init_program])[0], C)
        for l in range(L):
            all_inspirations = top.select(N)
            all_prompts, all_weights = [], []
            for inspirations in all_inspirations:
                all_prompts += [task.prompts([top.responses[s] for s in inspirations])] * K
                all_weights += [rms_weights([top.rewards[s] for s in inspirations])] * K
            responses = self.pipeline.group_generate(all_prompts, all_weights)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            top.update(all_inspirations, [responses[c * K:(c + 1) * K] for c in range(C)], rewards.view(C, K).tolist())

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
