"""Searches config.task.name's programs toward a higher reward by sequential conditioned sampling (SCS, Gideoni et al.,
2026) guided by a mixture of config.N distributions, with config.model.name inside SGLang, no training, in config.C
independent runs.

Run c's dataset is its responses of finite reward of its latest step that has any, x_j with rewards r_j, empty at first.
Every one of config.L steps samples config.K responses per run and evaluates each once. While a run's dataset is empty,
its responses are sampled to the task's prompts, as SCS's first generation. Otherwise the run has N distributions p_i,
shared by its K samples, p_i being the model under the task's prompts_with_rewards of its own min(config.E, dataset
size) in-context examples drawn at random from the dataset without replacement. With p_ij = p_i(x_j) normalized by the
T_j tokens of x_j, guide-new.md's expected reward of each distribution is

    q_ij = p_ij^(1/T_j) / Σ_k p_ik^(1/T_k),
    r̂_i = Σ_j q_ij r_j,

all equal, so uniform weights, when the dataset has fewer than two distinct rewards or N = 1. The pipeline's
group_generate samples each response, all C × K in one call, with r̂ as the distributions' utilities: each token from
the per-token mixture max(0, Σ_i w_i p_i), w being extend_sglang's weights(r̂), guide-new.md's w* maximizing Σ_i w_i r̂_i.
A run's step without a response of finite reward keeps its dataset.

Each step logs every reward and response to wandb, -inf and "" included, response idx belonging to run idx // K, the
objective evaluations counting all runs.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/scs-guide.py --config config/scs-guide.py
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

    def run(self):
        """Runs config.L steps of drawing each run's N distributions, estimating their r̂ on its dataset, guided
        sampling and evaluation."""
        task, L, K, N, E, C = self.task, self.config.L, self.config.K, self.config.N, self.config.E, self.config.C
        datasets = [[] for _ in range(C)]
        for l in range(L):
            all_prompts, all_utilities = [], []
            for c in range(C):
                if datasets[c]:
                    prompts = []
                    for i in range(N):
                        in_contexts = random.sample(datasets[c], min(E, len(datasets[c])))
                        prompts.append(task.prompts_with_rewards([response for response, _ in in_contexts], [reward for _, reward in in_contexts]))
                    r = torch.tensor([reward for _, reward in datasets[c]], dtype=torch.float64)
                    r_hat = torch.zeros(N, dtype=torch.float64)
                    if N > 1 and r.unique().numel() > 1:
                        log_P, T = self.pipeline.log_probs(prompts, [response for response, _ in datasets[c]])
                        q = (log_P / T).softmax(1)
                        r_hat = q @ r
                else:
                    prompts, r_hat = [task.prompts()], torch.zeros(1, dtype=torch.float64)
                all_prompts += [prompts] * K
                all_utilities += [r_hat.tolist()] * K

            responses = self.pipeline.group_generate(all_prompts, all_utilities)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            for c in range(C):
                responses_c, rewards_c = responses[c * K:(c + 1) * K], rewards[c * K:(c + 1) * K].tolist()
                datasets[c] = [(response, reward) for response, reward in zip(responses_c, rewards_c) if math.isfinite(reward)] or datasets[c]

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/scs-guide.py", "Guided sequential conditioned sampling configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
