"""Searches config.task.name's responses toward a higher reward through mixtures of in-context distributions with
config.model.name inside SGLang, no training.

First, batches of config.K responses to the task's prompt are sampled until config.N × config.E of them have a finite
reward; the N distributions are prompts_with_rewards of E of those each, drawn without replacement. Each of config.L
steps then samples config.K responses by Pipeline.group_generate, response j from ProductLogitProcessor's product
mixture of the N distributions with weights W_j drawn uniformly from the simplex, and evaluates each once. The estimated reward r̂_i of distribution i is the least squares fit of the step's
finite rewards r_j ≈ Σ_i w_ij r̂_i. The distribution of config.drop_policy, "worst" of the lowest r̂_i or "random", is
then replaced by prompts_with_rewards of config.E of the step's responses with a finite reward under
config.selection_policy, "best" of the highest rewards or "random". A step with fewer than config.E finite rewards keeps
its distributions.

Every batch logs every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python trainer/mix-distribution.py --config config/mix-distribution.py
"""

import sys

import numpy as np
import torch
import wandb
from absl import flags
from ml_collections import config_flags

from extend_sglang import ProductLogitProcessor
from trainer.base import BaseTrainer

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def select_in_contexts(self, rewards: np.ndarray) -> np.ndarray:
        """Indices of config.E of the finite rewards under config.selection_policy: those of the highest rewards for
        "best", drawn at random without replacement for "random".

        Args:
            rewards: (K) float, at least config.E of them finite

        Returns:
            (config.E) int, ascending

        Raises:
            NotImplementedError: config.selection_policy is neither "best" nor "random"
        """
        E = self.config.E
        finite_indices = np.flatnonzero(np.isfinite(rewards))
        if self.config.selection_policy == "best":
            selected = finite_indices[np.argsort(rewards[finite_indices], kind="stable")[-E:]]
        elif self.config.selection_policy == "random":
            selected = np.random.choice(finite_indices, size=E, replace=False)
        else:
            raise NotImplementedError(f"Selection policy {self.config.selection_policy} not implemented")
        return np.sort(selected)

    def drop_index(self, r_hat: np.ndarray) -> int:
        """The distribution to replace under config.drop_policy: that of the lowest r̂_i for "worst", one at random for
        "random".

        Args:
            r_hat: (N) float, each distribution's estimated reward

        Raises:
            NotImplementedError: config.drop_policy is neither "worst" nor "random"
        """
        if self.config.drop_policy == "worst":
            return int(np.argmin(r_hat))
        elif self.config.drop_policy == "random":
            return int(np.random.randint(len(r_hat)))
        raise NotImplementedError(f"Drop policy {self.config.drop_policy} not implemented")

    def log_batch(self, objective_evaluations: int, rewards: np.ndarray, responses: list[str]):
        """Logs a batch's rewards and responses to wandb.

        Args:
            objective_evaluations: int, responses evaluated so far, this batch's included
            rewards: (K) float
            responses: list (K) of str
        """
        rewards = torch.tensor(rewards, dtype=torch.float64)
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
        self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

    def run(self):
        """Samples to the task's prompt until the distributions can be built, then runs config.L steps of sampling
        config.K responses from mixtures of the distributions with random weights, evaluating them, and replacing the
        distribution of the drop policy."""
        task, L, K, N, E = self.task, self.config.L, self.config.K, self.config.N, self.config.E
        objective_evaluations = 0

        responses, rewards = [], np.empty(0)
        while np.isfinite(rewards).sum() < N * E:
            batch_responses = self.pipeline.generate([task.prompt()] * K)
            batch_rewards = np.array(task.evaluate(batch_responses))
            objective_evaluations += K
            self.log_batch(objective_evaluations, batch_rewards, batch_responses)
            responses += batch_responses
            rewards = np.concatenate([rewards, batch_rewards])
        in_context_indices = np.random.choice(np.flatnonzero(np.isfinite(rewards)), size=(N, E), replace=False)
        distributions = [task.prompts_with_rewards([responses[k] for k in indices], rewards[indices].tolist()) for indices in in_context_indices]

        for l in range(L):
            all_W = np.random.dirichlet(np.ones(N), size=K)
            responses = self.pipeline.group_generate([distributions] * K, all_W.tolist(), group_logit_processor=ProductLogitProcessor)
            rewards = np.array(task.evaluate(responses))
            finite = np.isfinite(rewards)
            if finite.sum() >= E:
                r_hat = np.linalg.lstsq(all_W[finite], rewards[finite], rcond=None)[0]
                dropped = self.drop_index(r_hat)
                selected = self.select_in_contexts(rewards)
                distribution = task.prompts_with_rewards([responses[j] for j in selected], rewards[selected].tolist())
                distributions = distributions[:dropped] + distributions[dropped + 1 :] + [distribution]
            objective_evaluations += K
            self.log_batch(objective_evaluations, rewards, responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/mix-distribution.py", "Mixture of in-context distributions configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
