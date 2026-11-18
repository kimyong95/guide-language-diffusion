"""Searches config.task.name's programs toward a higher reward by IID random sampling (IID RS, Gideoni et al., 2026)
with config.model.name inside SGLang, no training.

Samples config.M responses to the task's prompts under the pipeline's sampling parameters, all M in one generate call,
and evaluates each once. No response conditions on another.

Every reward and response is logged to wandb, -inf and "" included, the rewards one objective evaluation at a time so
the highest reward so far is a curve over 1, ..., M.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/iid-rs.py --config config/iid-rs.py
"""

import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from trainer.base import BaseTrainer

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def run(self):
        """Samples config.M responses to the task's prompts in one generate call, evaluates them, and logs each
        response's reward as its own objective evaluation, in sampling order."""
        task, M = self.task, self.config.M
        responses = self.pipeline.generate([task.prompt()] * M)
        rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)

        for m in range(M):
            self.log_rewards(objective_evaluations=m + 1, rewards=rewards[m:m + 1])
        self.log_responses(objective_evaluations=M, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/iid-rs.py", "IID random sampling configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
