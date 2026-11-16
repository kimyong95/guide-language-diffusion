"""Mutates config.task.name's seed program toward a higher reward by guided sampling of config.model.name inside SGLang,
no training.

Every one of config.L steps samples config.K responses guided by N distributions p_i, p_i being the model under the
task's prompts_with_scores of its in-context examples with their rewards. The S responses so far of finite reward, the
seed included, sorted by reward, highest first, and reduced to the first of each distinct reward, are split in turn
into groups of config.E: the first N = min(config.N, ⌊S / E⌋) groups are the in-context examples, all S as one group
while S < E.

The dataset is the responses of finite reward of the latest config.M steps, the seed not among them, x_j with rewards
r_j, less the low outliers: those of Bonferroni-corrected one-sided p-value n Φ((r_j - r̄) / s) below 0.05, n,
r̄ and s being the number, mean and standard deviation of the rewards, none when s is 0 or n < 2. With p_ij = p_i(x_j)
normalized by the T_j tokens of x_j, guide-new.md's expected reward of each distribution is

    q_ij = p_ij^(1/T_j) / Σ_k p_ik^(1/T_k),
    r̂_i = Σ_j q_ij r_j,

all equal, so uniform weights, when the dataset has fewer than two distinct rewards or N = 1. The pipeline's
group_generate samples each response with r̂ as the distributions' utilities: each token from the per-token mixture
max(0, Σ_i w_i p_i), w being extend_sglang's weights(r̂), guide-new.md's w* maximizing Σ_i w_i r̂_i.

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

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of picking the in-context examples, estimating r̂ on the dataset, guided sampling and evaluation."""
        task, L, K, M, E = self.task, self.config.L, self.config.K, self.config.M, self.config.E
        all_responses, all_rewards, all_steps = [task.init_program], task.evaluate([task.init_program]), [-math.inf]
        for l in range(L):
            distinct = {}
            for s in sorted(range(len(all_rewards)), key=all_rewards.__getitem__, reverse=True):
                distinct.setdefault(all_rewards[s], s)
            ranked = list(distinct.values())
            N = min(self.config.N, max(1, len(ranked) // E))
            all_in_contexts = [ranked[i * E:(i + 1) * E] for i in range(N)]
            prompts = [task.prompts_with_scores([all_responses[s] for s in in_contexts], [all_rewards[s] for s in in_contexts]) for in_contexts in all_in_contexts]

            dataset = [s for s, step in enumerate(all_steps) if step >= l - M]
            r = torch.tensor([all_rewards[s] for s in dataset], dtype=torch.float64)
            if len(dataset) > 1 and r.std() > 0:
                kept = len(r) * torch.special.ndtr((r - r.mean()) / r.std()) >= 0.05
                dataset, r = [s for s, kept_j in zip(dataset, kept.tolist()) if kept_j], r[kept]
            r_hat = torch.zeros(N, dtype=torch.float64)
            if N > 1 and r.unique().numel() > 1:
                log_P, T = self.pipeline.log_probs(prompts, [all_responses[s] for s in dataset])
                q = (log_P / T).softmax(1)
                r_hat = q @ r

            responses = self.pipeline.group_generate([prompts] * K, [r_hat.tolist()] * K, "sum")
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            for response, reward in zip(responses, rewards.tolist()):
                if math.isfinite(reward):
                    all_responses.append(response)
                    all_rewards.append(reward)
                    all_steps.append(l)

            objective_evaluations = (l + 1) * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
