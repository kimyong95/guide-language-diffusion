"""Searches config.task.name's responses toward a higher reward through mixtures of in-context distributions with
config.model.name inside SGLang, no training.

First, batches of config.M responses to the task's prompt are sampled until config.N × config.E of them have a finite
reward, neither logged nor counted as objective evaluations; the N distributions are prompts_with_rewards of E of those
each, drawn without replacement. Each of config.L steps then samples config.M responses by Pipeline.group_generate,
response j from ProductLogitProcessor's product mixture of the N distributions with weights W_j, rewards_to_weights of
rewards drawn for each response under config.reward_policy, and evaluates each once:
    "random": from N(0, I_N)
    "random-exp": exp(r), r drawn from N(0, I_N)
    "random-flip-exp": -exp(-r), r drawn from N(0, I_N)
    "measure": from N(μ_i, σ_i²) for each distribution i, μ_i and σ_i² the mean and variance of its finite rewards of
        config.M responses sampled to it alone when it is built, neither logged nor counted as objective evaluations
The estimated reward r̂_i of distribution i is the least squares fit of the step's finite rewards r_j ≈ Σ_i w_ij r̂_i.
The incoming distribution, prompts_with_rewards of the config.E of the step's responses with the highest finite rewards,
then replaces the distribution of the lowest r̂_i; under "measure", only when its μ exceeds the highest μ_i of the N
distributions. A step with fewer than config.E finite rewards keeps its distributions.

Responses are sampled under the model's default sampling parameters, or under RAW_SAMPLING_PARAMS when
config.model.sampling_params is "raw": temperature 1, without top-k, top-p, min-p or penalties.

Every step logs every reward and response to wandb, -inf and "" included.

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
RAW_SAMPLING_PARAMS = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}


class Trainer(BaseTrainer):

    def setup_pipeline(self, **kwargs):
        """BaseTrainer's pipeline, under RAW_SAMPLING_PARAMS when config.model.sampling_params is "raw".

        Raises:
            NotImplementedError: config.model.sampling_params is neither "default" nor "raw"
        """
        super().setup_pipeline(**kwargs)
        if self.config.model.sampling_params == "raw":
            self.pipeline.sampling_params = RAW_SAMPLING_PARAMS
        elif self.config.model.sampling_params != "default":
            raise NotImplementedError(f"Sampling parameters {self.config.model.sampling_params} not implemented")

    @staticmethod
    def rewards_to_weights(rewards: np.ndarray) -> np.ndarray:
        """Weights of the N distributions of their rewards r, 1/N + (r - r̄) / σ_r: summing to 1, of unit variance, and
        possibly negative; uniform when the rewards are all equal.

        Args:
            rewards: (..., N) float

        Returns:
            (..., N) float
        """
        N = rewards.shape[-1]
        deviations, std = rewards - rewards.mean(-1, keepdims=True), rewards.std(-1, keepdims=True)
        return 1 / N + np.divide(deviations, std, out=np.zeros_like(deviations), where=std > 0)

    def measure_rewards(self, distributions: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of each distribution's finite rewards of config.M responses sampled to it alone, neither
        logged nor counted as objective evaluations.

        Args:
            distributions: list (G) of str

        Returns:
            reward_means: (G) float, nan for a distribution without a finite reward
            reward_variances: (G) float, nan for a distribution without a finite reward
        """
        M = self.config.M
        responses = self.pipeline.generate([distribution for distribution in distributions for _ in range(M)])
        rewards = np.array(self.task.evaluate(responses)).reshape(len(distributions), M)
        rewards = np.where(np.isfinite(rewards), rewards, np.nan)
        return np.nanmean(rewards, -1), np.nanvar(rewards, -1)

    def log_batch(self, objective_evaluations: int, rewards: np.ndarray, responses: list[str]):
        """Logs a batch's rewards and responses to wandb.

        Args:
            objective_evaluations: int, responses evaluated so far, this batch's included
            rewards: (M) float
            responses: list (M) of str
        """
        rewards = torch.tensor(rewards, dtype=torch.float64)
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
        self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

    def run(self):
        """Samples to the task's prompt until the distributions can be built, then runs config.L steps of sampling
        config.M responses from mixtures of the distributions with weights under config.reward_policy, evaluating them,
        and replacing the distribution of the lowest r̂_i.

        Raises:
            NotImplementedError: config.reward_policy is not "random", "random-exp", "random-flip-exp" or "measure"
        """
        task, L, M, N, E = self.task, self.config.L, self.config.M, self.config.N, self.config.E
        objective_evaluations = 0

        responses, rewards = [], np.empty(0)
        while np.isfinite(rewards).sum() < N * E:
            batch_responses = self.pipeline.generate([task.prompt()] * M)
            batch_rewards = np.array(task.evaluate(batch_responses))
            responses += batch_responses
            rewards = np.concatenate([rewards, batch_rewards])
        in_context_indices = np.random.choice(np.flatnonzero(np.isfinite(rewards)), size=(N, E), replace=False)
        distributions = [task.prompts_with_rewards([responses[k] for k in indices], rewards[indices].tolist()) for indices in in_context_indices]
        if self.config.reward_policy == "measure":
            reward_means, reward_variances = self.measure_rewards(distributions)

        for l in range(L):
            if self.config.reward_policy == "random":
                all_W = self.rewards_to_weights(np.random.randn(M, N))
            elif self.config.reward_policy == "random-exp":
                all_W = self.rewards_to_weights(np.exp(np.random.randn(M, N)))
            elif self.config.reward_policy == "random-flip-exp":
                all_W = self.rewards_to_weights(-np.exp(-np.random.randn(M, N)))
            elif self.config.reward_policy == "measure":
                all_W = self.rewards_to_weights(np.random.normal(reward_means, np.sqrt(reward_variances), size=(M, N)))
            else:
                raise NotImplementedError(f"Reward policy {self.config.reward_policy} not implemented")
            responses = self.pipeline.group_generate([distributions] * M, all_W.tolist(), group_logit_processor=ProductLogitProcessor)
            rewards = np.array(task.evaluate(responses))
            finite = np.isfinite(rewards)
            if finite.sum() >= E:
                finite_indices = np.flatnonzero(finite)
                selected = np.sort(finite_indices[np.argsort(rewards[finite_indices], kind="stable")[-E:]])
                distribution = task.prompts_with_rewards([responses[j] for j in selected], rewards[selected].tolist())
                if self.config.reward_policy == "measure":
                    reward_mean, reward_variance = self.measure_rewards([distribution])
                if self.config.reward_policy != "measure" or reward_mean[0] > reward_means.max():
                    r_hat = np.linalg.lstsq(all_W[finite], rewards[finite], rcond=None)[0]
                    dropped = int(np.argmin(r_hat))
                    distributions = distributions[:dropped] + distributions[dropped + 1 :] + [distribution]
                    if self.config.reward_policy == "measure":
                        reward_means = np.concatenate([np.delete(reward_means, dropped), reward_mean])
                        reward_variances = np.concatenate([np.delete(reward_variances, dropped), reward_variance])
            objective_evaluations += M
            self.log_batch(objective_evaluations, rewards, responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/mix-distribution.py", "Mixture of in-context distributions configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
