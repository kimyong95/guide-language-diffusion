"""Mutates config.task.name's seed program toward a higher reward as agent.py does, its N = 3 in-context examples turned
into the guides of guide.py, no training.

Every one of config.L steps samples each chain once: selector.py's RPUCG picks up to N of the chain's nodes as its
inspirations, best first, and config.K new responses are sampled independently by guide.py's guided sampling, under
log p_2 + log p_1 - log p_3 with three inspirations and log p_1 with fewer, p_i being the model under the task's
prompts_with_scores of inspiration i alone with its reward. RPUCG is updated as in agent.py, and each step logs every
reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/agent-guide.py --config config/agent-guide.py
"""

import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from base import BaseTrainer
from guide import N, guide_weights
from selector import RPUCG

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of RPUCG selection, guided sampling, evaluation and update from the seed program."""
        task, L, C, K = self.task, self.config.L, self.config.C, self.config.K
        rpucg = RPUCG(task.init_program, task.evaluate([task.init_program])[0], C)
        for l in range(L):
            all_inspirations = rpucg.select(N)
            all_prompts, all_weights = [], []
            for inspirations in all_inspirations:
                all_prompts += [[task.prompts_with_scores([rpucg.responses[s]], [rpucg.rewards[s]]) for s in inspirations]] * K
                all_weights += [guide_weights(len(inspirations))] * K
            responses = self.pipeline.group_generate(all_prompts, all_weights)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            rpucg.update(all_inspirations, [responses[c * K:(c + 1) * K] for c in range(C)], rewards.view(C, K).tolist())

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/agent-guide.py", "Agent-guide configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
