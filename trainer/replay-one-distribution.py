"""Searches config.task.name's responses toward a higher reward by moving a single in-context distribution with
config.model.name inside SGLang, no mixture and no training.

The base distribution p_base starts as the task's prompt. Each of config.L steps samples config.M responses from it and
evaluates each once. Every combination of config.E of the step's config.K distinct responses of the highest finite
rewards is an eval distribution p_i, prompts_with_rewards of them. On the step's finite responses x_j, each eval
distribution's own config.E in-context responses excluded from its fit, guide.md M fits each eval distribution's β_i by
its Newton update from β_i = 0:
    β_i ← β_i + (E_{q_i}[r] - E_{p_i}[r]) / Var_{p_i}[r]
where q_ij ∝ p_i(x_j) / p_base(x_j) and p_i(x_j; β_i) ∝ exp(β_i r_j), all exact sequence likelihoods under the base
distribution and eval distributions. The base distribution's own β is 0, so at the end of the step the eval
distribution of the highest β becomes the base distribution only when that β is positive.

Responses are sampled under RAW_SAMPLING_PARAMS: temperature 1, without top-k, top-p, min-p or penalties, so that
p_base(x_j) is the exact likelihood of sampling x_j.

Every step logs every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python trainer/replay-one-distribution.py --config config/replay-one-distribution.py
"""

import itertools
import sys

import numpy as np
import torch
import wandb
from absl import flags
from ml_collections import config_flags
from scipy.special import logsumexp

from trainer.base import BaseTrainer

FLAGS = flags.FLAGS
RAW_SAMPLING_PARAMS = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}


class Trainer(BaseTrainer):

    def setup_pipeline(self, **kwargs):
        """BaseTrainer's pipeline, sampling under RAW_SAMPLING_PARAMS."""
        super().setup_pipeline(**kwargs)
        self.pipeline.sampling_params = RAW_SAMPLING_PARAMS

    @staticmethod
    def fit_betas(log_p: np.ndarray, log_p_base: np.ndarray, rewards: np.ndarray, mask: np.ndarray, iterations: int = 100) -> np.ndarray:
        """guide.md's β_i of each distribution i, by its Newton update from β_i = 0:
        β_i ← β_i + (E_{q_i}[r] - E_{p_i}[r]) / Var_{p_i}[r], with q_ij ∝ p_i(x_j) / p_base(x_j) and
        p_i(x_j; β_i) ∝ exp(β_i r_j) over the x_j of mask_i. An update stops once Var_{p_i}[r] is 0.

        Args:
            log_p: (N, M) float, log p_i(x_j)
            log_p_base: (M,) float, log p_base(x_j)
            rewards: (M,) float, r_j
            mask: (N, M) bool, whether x_j enters distribution i's fit, True for at least one j per i
            iterations: int, Newton updates

        Returns:
            (N,) float
        """
        log_q = np.where(mask, log_p - log_p_base, -np.inf)
        q = np.exp(log_q - logsumexp(log_q, axis=1, keepdims=True))
        E_q = q @ rewards
        betas = np.zeros(len(log_p))
        for _ in range(iterations):
            log_p_beta = np.where(mask, betas[:, None] * rewards, -np.inf)
            p_beta = np.exp(log_p_beta - logsumexp(log_p_beta, axis=1, keepdims=True))
            E_p = p_beta @ rewards
            Var_p = (p_beta * (rewards - E_p[:, None]) ** 2).sum(1)
            betas = betas + np.divide(E_q - E_p, Var_p, out=np.zeros_like(E_q), where=Var_p > 0)
        return betas

    def run(self):
        """Runs config.L steps of sampling config.M responses from the base distribution, evaluating them, fitting β on
        them, and replacing the base distribution by the eval distribution of the highest β when it is positive."""
        task, L, M, K, E = self.task, self.config.L, self.config.M, self.config.K, self.config.E
        base_distribution = task.prompt()

        for l in range(L):
            responses = self.pipeline.generate([base_distribution] * M)
            rewards = np.array(task.evaluate(responses), dtype=np.float64)
            finite_indices = np.flatnonzero(np.isfinite(rewards))

            top, top_stories = [], set()
            for j in finite_indices[np.argsort(-rewards[finite_indices], kind="stable")]:
                if responses[j].strip() not in top_stories:
                    top.append(int(j))
                    top_stories.add(responses[j].strip())
                if len(top) == K:
                    break

            eval_distributions, betas = [], np.zeros(0)
            if len(top) >= E and len(finite_indices) >= E + 2:
                all_in_context_indices = [sorted(indices) for indices in itertools.combinations(top, E)]
                eval_distributions = [task.prompts_with_rewards([responses[j] for j in indices], rewards[indices].tolist()) for indices in all_in_context_indices]
                mask = np.array([[j not in indices for j in finite_indices] for indices in all_in_context_indices])
                log_p = self.pipeline.log_probs(eval_distributions + [base_distribution], [responses[j] for j in finite_indices]).numpy()
                betas = self.fit_betas(log_p[:-1], log_p[-1], rewards[finite_indices], mask)

            objective_evaluations = (l + 1) * M
            rewards = torch.tensor(rewards, dtype=torch.float64)
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

            if len(betas) and betas.max() > 0:
                base_distribution = eval_distributions[int(np.argmax(betas))]

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/replay-one-distribution.py", "Single in-context distribution configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
