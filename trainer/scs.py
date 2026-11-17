"""Searches config.task.name's programs toward a higher reward by sequential conditioned sampling (SCS, Gideoni et al.,
2026) with config.model.name inside SGLang, no training.

The dataset is the responses of finite reward of the latest step that has any, with their rewards, empty at first.
Every one of config.L steps samples config.K responses under the pipeline's sampling parameters, all K in one generate
call, and evaluates each once: each to the task's prompts while the dataset is empty, otherwise to its
prompts_with_rewards of up to config.E in-context examples selected from the dataset by config.selection_policy,
"random" as the paper does, "best" or "best+random". A step without a response of finite reward keeps the dataset, so
the next step selects from it again, as the paper retries such a generation.

Each step logs every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/scs.py --config config/scs.py
"""

import math
import random
import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from trainer.base import BaseTrainer

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def select_in_contexts(self, dataset: list) -> list:
        """The dataset's in-context examples under config.selection_policy: config.E of them drawn at random without
        replacement for "random", those of the config.E highest rewards, highest first, for "best", or the one of the
        highest reward followed by config.E - 1 of the rest at random for "best+random". An earlier example comes first
        on ties.

        Args:
            dataset: list (D) of (str, float), the responses with their rewards, nonempty

        Returns:
            list (min(config.E, D)) of (str, float)

        Raises:
            NotImplementedError: config.selection_policy is none of "random", "best" and "best+random"
        """
        E = self.config.E
        if self.config.selection_policy == "random":
            return random.sample(dataset, min(E, len(dataset)))
        elif self.config.selection_policy == "best":
            return sorted(dataset, key=lambda item: item[1], reverse=True)[:E]
        elif self.config.selection_policy == "best+random":
            dataset = sorted(dataset, key=lambda item: item[1], reverse=True)
            best = dataset.pop(0)
            return [best] + random.sample(dataset, min(E - 1, len(dataset)))
        else:
            raise NotImplementedError(f"Selection policy {self.config.selection_policy} not implemented")

    def run(self):
        """Runs config.L steps of sampling config.K responses conditioned on the dataset, evaluating them and replacing
        the dataset."""
        task, L, K = self.task, self.config.L, self.config.K
        dataset = []
        for l in range(L):
            prompts = []
            for k in range(K):
                if dataset:
                    in_contexts = self.select_in_contexts(dataset)
                    prompts.append(task.prompts_with_rewards([response for response, _ in in_contexts], [reward for _, reward in in_contexts]))
                else:
                    prompts.append(task.prompts())
            responses = self.pipeline.generate(prompts)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            dataset = [(response, reward) for response, reward in zip(responses, rewards.tolist()) if math.isfinite(reward)] or dataset

            objective_evaluations = (l + 1) * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/scs.py", "Sequential conditioned sampling configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
