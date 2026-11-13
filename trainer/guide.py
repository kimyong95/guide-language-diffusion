"""Mutates config.task.name's seed program toward a higher reward by guided sampling of config.model.name inside SGLang,
no training.

Of the responses so far of finite reward, the seed at the start, the best config.M are the data D, responses x_j with
rewards r_j. Every one of config.L steps samples config.K responses guided by up to config.N responses drawn uniformly
from D without replacement, p_i being the model under the task's prompts of response i alone. With P_ij = p_i(x_j)
normalized by the T_j tokens of x_j and the base the mixture (1/N) Σ_k p_k, guide-algorithm.md's β is

    log q_ij = log P_ij / T_j - log Σ_k P_kj^(1/T_j), normalized over j,
    β_i = (r - r̄)⁺ log q_i,

⁺ being the Moore–Penrose pseudoinverse: β_i is the least-squares slope of log q_ij on r_j, 0 when the rewards are all
equal. The pipeline's group_generate samples each response with β as the guides' utilities, so under extend_sglang's
weights(β), its negative guidance clipped. The first step, of N = M = 1, samples the seed's prompt alone.

Each step logs every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/guide.py --config config/guide.py
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
        """Runs config.L steps of drawing the guides from D, estimating β, guided sampling and evaluation."""
        task, L, K, M, N = self.task, self.config.L, self.config.K, self.config.M, self.config.N
        all_responses, all_rewards = [task.init_program], task.evaluate([task.init_program])
        for l in range(L):
            D = sorted(range(len(all_rewards)), key=all_rewards.__getitem__, reverse=True)[:M]
            prompts = task.prompts([all_responses[s] for s in random.sample(D, min(N, len(D)))])
            log_P, T = self.pipeline.log_probs(prompts, [all_responses[s] for s in D])
            log_P = log_P / T
            log_q = (log_P - log_P.logsumexp(0)).log_softmax(1)
            r = torch.tensor([all_rewards[s] for s in D], dtype=torch.float64)
            beta = (torch.linalg.pinv((r - r.mean())[:, None]) @ log_q.T)[0]

            responses = self.pipeline.group_generate([prompts] * K, [beta.tolist()] * K)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            for response, reward in zip(responses, rewards.tolist()):
                if math.isfinite(reward):
                    all_responses.append(response)
                    all_rewards.append(reward)

            objective_evaluations = (l + 1) * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
