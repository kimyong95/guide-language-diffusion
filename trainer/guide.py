"""Mutates config.task.name's seed program toward a higher reward by guided sampling of config.model.name inside SGLang,
its guides picked by selector.py's Rpucg over config.C chains, no training.

Every one of config.L steps samples each chain once: Rpucg picks up to config.G of the chain's nodes as its
inspirations, best first, and config.K new responses are sampled independently under a weighted sum of log-probabilities over them,
p_i being the model under the task's prompts of inspiration i alone. Every guide thinks at config.model.reasoning_effort
when config.model.enable_thinking, scoring the response after its own prompt. A response's answer is what follows its
</think> when thinking, empty when its thinking never ends, and the whole response otherwise. Every response is
sampled under Σ_i w_i log p_i, the weights maximising Σ_i w_i r_i with Σ_i w_i = 1 and the mean w_i² at most 1, r_i
being the inspirations' rewards; uniform when they are all equal.

The C × K responses are sampled together by the pipeline's group_generate, each as one group of requests, one per guide.
Each token of a response is sampled from its weighted sum under the model's generation config, as the pipeline's generate
samples, and given to all of the group's requests, so its guides advance and stop together; a response fills
config.model.max_total_tokens with its group's longest prompt at most.

Each response's reward r is the task's evaluate, config.task.concurrency at a time, and every response joins the population
by Rpucg's update, one with a non-finite reward never an inspiration.
Each step logs every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/guide.py --config config/guide.py
"""

import math
import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from trainer.base import BaseTrainer
from trainer.selector import Rpucg

FLAGS = flags.FLAGS


def weights(rewards):
    """The weights maximising Σ_i w_i r_i with Σ_i w_i = 1 and the mean w_i² at most 1; uniform when the rewards are all
    equal.

    Args:
        rewards: list (G) of float, finite, each guide's reward

    Returns:
        list (G) of float, 1/G + √(G - 1/G) (r_i - r̄) / √(Σ_j (r_j - r̄)²), r̄ being the rewards' mean
    """
    G = len(rewards)
    if min(rewards) == max(rewards):
        return [1 / G] * G
    mean = sum(rewards) / G
    norm = math.sqrt(sum((r - mean) ** 2 for r in rewards))
    return [1 / G + math.sqrt(G - 1 / G) * (r - mean) / norm for r in rewards]


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of Rpucg selection, guided sampling, evaluation and update from the seed program."""
        task, L, C, K, G = self.task, self.config.L, self.config.C, self.config.K, self.config.G
        selector = Rpucg(task.init_program, task.evaluate([task.init_program])[0], C)
        for l in range(L):
            all_inspirations = selector.select(G)
            all_prompts, all_weights = [], []
            for inspirations in all_inspirations:
                all_prompts += [task.prompts([selector.responses[s] for s in inspirations])] * K
                all_weights += [weights([selector.rewards[s] for s in inspirations])] * K
            responses = self.pipeline.group_generate(all_prompts, all_weights)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            selector.update(all_inspirations, [responses[c * K:(c + 1) * K] for c in range(C)], rewards.view(C, K).tolist())

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
