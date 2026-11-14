"""Mutates config.task.name's seed program toward a higher reward by guided sampling of config.model.name inside SGLang,
no training.

Every one of config.L steps samples config.K responses guided by N distributions p_i, p_i being the model under the
task's prompts_with_scores of its in-context examples with their rewards. The S responses so far of finite reward, the
seed included, sorted by reward, highest first, and reduced to the first of each distinct reward, are split in turn
into groups of config.E: the first N = min(config.N, ⌊S / E⌋) groups are the in-context examples, all S as one group
while S < E.

The dataset is the responses of finite reward of the latest config.M steps, the seed not among them, x_j with rewards
r_j, less the low outliers: those of Bonferroni-corrected one-sided p-value n Φ((r_j - r̄) / s) below 0.05, n,
r̄ and s being the number, mean and standard deviation of the rewards, none when s is 0 or n < 2. With P_ij = p_i(x_j)
normalized by the T_j tokens of x_j and the base the mixture (1/N) Σ_k p_k, guide-algorithm.md's β is

    log q_ij = log P_ij / T_j - log Σ_k P_kj^(1/T_j), normalized over j,
    β_i = argmin_β KL(q_i, p_i(·; β)),

solved by fit_beta, 0 when the dataset has fewer than two distinct rewards or N = 1. The pipeline's group_generate
samples each response with β as the distributions' utilities, so under extend_sglang's weights(β).

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


def fit_beta(log_q, r):
    """β of guide-algorithm.md: each β_i minimizing KL(q_i, p_i(·; β_i)), p_i(x_j; β_i) ∝ exp(β_i r_j), by its Newton
    update from 0 until every gradient Σ_j q_ij r_j - Σ_j p_i(x_j; β_i) r_j is within 1e-10 of the rewards' range.

    Args:
        log_q: (N, n) float64, log q_ij, each row normalized over j
        r: (n) float64, of at least two distinct values

    Returns:
        (N) float64

    Raises:
        RuntimeError: the Newton updates diverge
    """
    q = log_q.exp()
    tolerance = 1e-10 * (r.max() - r.min())
    beta = torch.zeros(len(q), dtype=r.dtype)
    while True:
        p = (beta[:, None] * r).softmax(1)
        mean = p @ r
        gradient = q @ r - mean
        if gradient.abs().max() <= tolerance:
            return beta
        beta = beta + gradient / (p @ r**2 - mean**2)
        if not beta.isfinite().all():
            raise RuntimeError(f"Newton updates of β diverged: {beta.tolist()}")


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of picking the in-context examples, estimating β on the dataset, guided sampling and evaluation."""
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
            beta = torch.zeros(N, dtype=torch.float64)
            if N > 1 and r.unique().numel() > 1:
                log_P, T = self.pipeline.log_probs(prompts, [all_responses[s] for s in dataset])
                log_P = log_P / T
                log_q = (log_P - log_P.logsumexp(0)).log_softmax(1)
                beta = fit_beta(log_q, r)

            responses = self.pipeline.group_generate([prompts] * K, [beta.tolist()] * K)
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
