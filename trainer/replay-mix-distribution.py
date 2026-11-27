"""Searches config.task.name's responses toward a higher reward by moving a mixture of in-context distributions with
config.model.name inside SGLang, no training.

The base distribution p_base starts as the task's prompt. Each of config.L steps samples config.M responses from it and
evaluates each once. Every combination of config.E of the step's config.K distinct responses of the highest finite
rewards is an eval distribution p_i, prompts_with_rewards of them. On the step's finite responses x_j, each eval
distribution's own config.E in-context responses excluded from its fit, guide.md N fits the weights of the eval
distributions' mixture π(x; w) ∝ Π_i p_i(x)^{w_i} over at most config.n distributions:
    w*_i = 1/n + √(n - 1/n) (g_i - ḡ_S) / ‖g̃_S‖ on S*, 0 elsewhere, g_i = Σ_j φ̃_ij r̃_ij / Σ_j r̃_ij²
of φ_i(x_j) = log p_i(x_j) / p_base(x_j) and r_j, each centered over distribution i's x_j, all sequence likelihoods,
where S* is the head-and-tail subset of the smallest and largest g_i maximizing ḡ_S + √(n - 1/n) ‖g̃_S‖.
The base distribution's own β is 0, so at the end of the step the mixture π(x; w*) of S* becomes the base distribution
only when its β(w*) = gᵀw* is positive.

The base distribution is the mixture π(x; w) of its distributions p_k, so log p_base(x_j) = Σ_k w_k log p_k(x_j) up to
log Z_w, which the centering cancels. It is sampled by ProductLogitProcessor, each token from Π_k p_k(y_t | y_<t)^{w_k}
renormalized, under RAW_SAMPLING_PARAMS: temperature 1, without top-k, top-p, min-p or penalties, as log_probs scores.

Every step logs every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python trainer/replay-mix-distribution.py --config config/replay-mix-distribution.py
"""

import itertools
import sys

import numpy as np
import torch
import wandb
from absl import flags
from ml_collections import config_flags

from extend_sglang import ProductLogitProcessor
from trainer.base import BaseTrainer

FLAGS = flags.FLAGS
RAW_SAMPLING_PARAMS = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}


class Trainer(BaseTrainer):

    def setup_pipeline(self, **kwargs):
        """BaseTrainer's pipeline, sampling under RAW_SAMPLING_PARAMS."""
        super().setup_pipeline(**kwargs)
        self.pipeline.sampling_params = RAW_SAMPLING_PARAMS

    @staticmethod
    def fit_weights(phi: np.ndarray, rewards: np.ndarray, mask: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float]:
        """guide.md N's sparse w* maximizing β(w) = gᵀw subject to 1ᵀw = 1, ‖w‖² ≤ n and ‖w‖₀ ≤ n, with
        g_i = Σ_j φ̃_ij r̃_ij / Σ_j r̃_ij² over the x_j of mask_i, φ̃_i and r̃_i centered over them. Its support S* is the
        head-and-tail subset S_k of the k smallest and n - k largest g_i maximizing ḡ_S + √(n - 1/n) ‖g̃_S‖, and on it
        w*_i = 1/n + √(n - 1/n) (g_i - ḡ_S) / ‖g̃_S‖. w* is uniform on S* when g̃_S is 0, and g_i is 0 when r̃_i is.

        Args:
            phi: (N, M) float, φ_i(x_j) = log p_i(x_j) / p_base(x_j)
            rewards: (M,) float, r_j
            mask: (N, M) bool, whether x_j enters distribution i's fit, True for at least one j per i
            n: int, at most N

        Returns:
            (n,) int, S*
            (n,) float, w* on S*
            float, β(w*) = gᵀw*
        """
        N = len(phi)
        count = mask.sum(1, keepdims=True)
        phi_tilde = np.where(mask, phi - np.where(mask, phi, 0).sum(1, keepdims=True) / count, 0)
        r_tilde = np.where(mask, rewards - np.where(mask, rewards, 0).sum(1, keepdims=True) / count, 0)
        r_norm2 = (r_tilde ** 2).sum(1)
        g = np.divide((phi_tilde * r_tilde).sum(1), r_norm2, out=np.zeros(N), where=r_norm2 > 0)
        order = np.argsort(g, kind="stable")
        all_S = [np.concatenate([order[:k], order[N - n + k:]]) for k in range(n + 1)]
        S = max(all_S, key=lambda S: g[S].mean() + np.sqrt(n - 1 / n) * np.linalg.norm(g[S] - g[S].mean()))
        g_tilde = g[S] - g[S].mean()
        g_norm = np.linalg.norm(g_tilde)
        w = np.full(n, 1 / n) + (np.sqrt(n - 1 / n) * g_tilde / g_norm if g_norm > 0 else 0)
        return S, w, float(g[S] @ w)

    def run(self):
        """Runs config.L steps of sampling config.M responses from the base distribution, evaluating them, fitting w* on
        them, and replacing the base distribution by the eval distributions' mixture π(x; w*) of at most config.n
        distributions when its β(w*) is positive."""
        task, L, M, K, n, E = self.task, self.config.L, self.config.M, self.config.K, self.config.n, self.config.E
        base_distributions, w = [task.prompt()], np.ones(1)

        for l in range(L):
            responses = self.pipeline.group_generate([base_distributions] * M, [w.tolist()] * M, group_logit_processor=ProductLogitProcessor)
            rewards = np.array(task.evaluate(responses), dtype=np.float64)
            finite_indices = np.flatnonzero(np.isfinite(rewards))

            top, top_stories = [], set()
            for j in finite_indices[np.argsort(-rewards[finite_indices], kind="stable")]:
                if responses[j].strip() not in top_stories:
                    top.append(int(j))
                    top_stories.add(responses[j].strip())
                if len(top) == K:
                    break

            eval_distributions, beta = [], 0.0
            if len(top) >= E and len(finite_indices) >= E + 2:
                all_in_context_indices = [sorted(indices) for indices in itertools.combinations(top, E)]
                eval_distributions = [task.prompts_with_rewards([responses[j] for j in indices], rewards[indices].tolist()) for indices in all_in_context_indices]
                mask = np.array([[j not in indices for j in finite_indices] for indices in all_in_context_indices])
                log_p = self.pipeline.log_probs(eval_distributions + base_distributions, [responses[j] for j in finite_indices]).numpy()
                N = len(eval_distributions)
                log_p_base = w @ log_p[N:]
                S, w_star, beta = self.fit_weights(log_p[:N] - log_p_base, rewards[finite_indices], mask, min(n, N))

            objective_evaluations = (l + 1) * M
            rewards = torch.tensor(rewards, dtype=torch.float64)
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

            if beta > 0:
                base_distributions, w = [eval_distributions[i] for i in S], w_star

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/replay-mix-distribution.py", "Mixture of in-context distributions configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
