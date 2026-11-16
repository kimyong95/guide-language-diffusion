"""Searches config.task.name's programs toward a higher reward by sequential conditioned sampling (SCS, Gideoni et al.,
2026) with config.model.name inside SGLang, no training, in config.C independent runs.

Run c's dataset is its responses of finite reward of its latest step that has any, with their rewards, empty at first.
Every one of config.L steps samples config.K responses per run under the pipeline's sampling parameters, all C × K in
one generate call, and evaluates each once: each to the task's prompts while its run's dataset is empty, otherwise to
its prompts_with_rewards of up to config.E in-context examples selected from that dataset by config.selection_policy,
"random" as the paper does or "greedy". A run's step without a response of finite reward keeps its dataset, so its next
step selects from it again, as the paper retries such a generation.

Each step logs every reward and response to wandb, -inf and "" included, response idx belonging to run idx // K, the
objective evaluations counting all runs.

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
        replacement, or those of the config.E highest distinct rewards, highest first, an earlier one kept on ties.

        Args:
            dataset: list (D) of (str, float), a run's responses with their rewards, nonempty

        Returns:
            list (min(config.E, D) for "random", at most config.E for "greedy") of (str, float)

        Raises:
            NotImplementedError: config.selection_policy is neither "random" nor "greedy"
        """
        E = self.config.E
        if self.config.selection_policy == "random":
            return random.sample(dataset, min(E, len(dataset)))
        elif self.config.selection_policy == "greedy":
            distinct = {}
            for response, reward in sorted(dataset, key=lambda item: item[1], reverse=True):
                distinct.setdefault(reward, response)
            return [(response, reward) for reward, response in list(distinct.items())[:E]]
        else:
            raise NotImplementedError(f"Selection policy {self.config.selection_policy} not implemented")

    def run(self):
        """Runs config.L steps of sampling config.K responses per run conditioned on its dataset, evaluating them and
        replacing the datasets."""
        task, L, K, C = self.task, self.config.L, self.config.K, self.config.C
        datasets = [[] for _ in range(C)]
        for l in range(L):
            prompts = []
            for c in range(C):
                for k in range(K):
                    if datasets[c]:
                        in_contexts = self.select_in_contexts(datasets[c])
                        prompts.append(task.prompts_with_rewards([response for response, _ in in_contexts], [reward for _, reward in in_contexts]))
                    else:
                        prompts.append(task.prompts())
            responses = self.pipeline.generate(prompts)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            for c in range(C):
                responses_c, rewards_c = responses[c * K:(c + 1) * K], rewards[c * K:(c + 1) * K].tolist()
                datasets[c] = [(response, reward) for response, reward in zip(responses_c, rewards_c) if math.isfinite(reward)] or datasets[c]

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/scs.py", "Sequential conditioned sampling configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
