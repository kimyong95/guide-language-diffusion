"""Mutates config.task.name's seed program toward a higher reward as guide.py does, each guide's prompt holding config.N
in-context inspirations instead of one, no training.

Every one of config.L steps samples each chain once: Top picks up to config.G combinations of config.N of the chain's
population of distinct rewards, those of the highest average reward, best first, and config.K new responses are sampled
independently under Σ_i w_i log p_i, p_i being the model under the task's prompts_with_scores of combination i with its
rewards, and w_i guide.py's rms_weights of the combinations' average rewards. Sampling, evaluation and logging are as
in guide.py.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/agent-guide.py --config config/agent-guide.py
"""

import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from base import BaseTrainer
from guide import rms_weights
from selector import Top

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of Top selection, guided sampling, evaluation and update from the seed program."""
        task, L, C, K, G, N = self.task, self.config.L, self.config.C, self.config.K, self.config.G, self.config.N
        top = Top(task.init_program, task.evaluate([task.init_program])[0], C)
        for l in range(L):
            all_guides = top.select(G, N)
            all_prompts, all_weights = [], []
            for guides in all_guides:
                all_prompts += [[task.prompts_with_scores([top.responses[s] for s in inspirations], [top.rewards[s] for s in inspirations]) for inspirations in guides]] * K
                all_weights += [rms_weights([sum(top.rewards[s] for s in inspirations) / len(inspirations) for inspirations in guides])] * K
            responses = self.pipeline.group_generate(all_prompts, all_weights)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            top.update(all_guides, [responses[c * K:(c + 1) * K] for c in range(C)], rewards.view(C, K).tolist())

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/agent-guide.py", "Agent-guide configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
