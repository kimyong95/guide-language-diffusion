"""Mutates config.task.name's seed program toward a higher reward by guided sampling of config.model.name inside SGLang,
its guides picked by selector.py's Rpucg over config.C chains, no training.

Every one of config.L steps samples each chain once: Rpucg picks up to config.G of the chain's nodes as its
in-context examples, best first, and config.K new responses are sampled independently under a weighted sum of
log-probabilities over them, p_i being the model under the task's prompts of in-context example i alone. Every guide
thinks at config.model.reasoning_effort when config.model.enable_thinking, scoring the response after its own prompt. A
response's answer is what follows its </think> when thinking, empty when its thinking never ends, and the whole
response otherwise. Every response is sampled under the uniform mix (1/G) Σ_i log p_i of its G guides as base, guided
by g_i = w_i - 1/G, its negative guidance summed then clipped to lower probabilities only: w being extend_sglang's
weights of the guides' utilities u_i, maximising Σ_i w_i u_i with Σ_i w_i = 1 and the mean w_i² at most 1, a guide's
utility its in-context example's reward.

The C × K responses are sampled together by the pipeline's group_generate, each as one group of requests, one per guide.
Each token of a response is sampled from its guided log probabilities under the model's generation config, as the
pipeline's generate samples, and given to all of the group's requests, so its guides advance and stop together; a
response fills config.model.max_total_tokens with its group's longest prompt at most.

Each response's reward r is the task's evaluate, config.task.concurrency at a time, and every response joins the population
by Rpucg's update, one with a non-finite reward never an in-context example.
Each step logs every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/guide.py --config config/guide.py
"""

import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from trainer.base import BaseTrainer
from trainer.selector import Rpucg

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of Rpucg selection, guided sampling, evaluation and update from the seed program."""
        task, L, C, K, G = self.task, self.config.L, self.config.C, self.config.K, self.config.G
        selector = Rpucg(task.init_program, task.evaluate([task.init_program])[0], C)
        for l in range(L):
            all_in_contexts = selector.select(G)
            all_prompts, all_utilities = [], []
            for in_contexts in all_in_contexts:
                all_prompts += [task.prompts([selector.responses[s] for s in in_contexts])] * K
                all_utilities += [[selector.rewards[s] for s in in_contexts]] * K
            responses = self.pipeline.group_generate(all_prompts, all_utilities)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            selector.update(all_in_contexts, [responses[c * K:(c + 1) * K] for c in range(C)], rewards.view(C, K).tolist())

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
