"""Mutates config.task.name's seed program toward a higher reward by in-context sampling of config.model.name inside
SGLang, its in-context examples picked over config.C chains by selector.py's RPUCG, no guidance and no training.

The population holds every response, starting as the task's seed program alone, taken as a response, the root of every
chain. Every one of config.L steps samples each chain once: the selector picks up to
config.N of the chain's nodes, and config.K new responses are sampled independently under the task's prompts_with_scores
of them with their rewards, best first, thinking at config.model.reasoning_effort when config.model.enable_thinking;
all C × K responses come from one generate call. A response's answer is what follows its </think> when thinking, empty
when its thinking never ends, and the whole response otherwise.

Each response's reward r is the task's evaluate, config.task.concurrency at a time. Every response joins the population
by the selector's update, one with a non-finite reward never an inspiration but counted in the ranks. Each step logs
every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/agent.py --config config/agent.py
"""

import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from base import BaseTrainer
from selector import RPUCG

FLAGS = flags.FLAGS


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of selection, in-context sampling, evaluation and update from the seed program."""
        task, L, C, K, N = self.task, self.config.L, self.config.C, self.config.K, self.config.N
        selector = RPUCG(task.init_program, task.evaluate([task.init_program])[0], C)
        for l in range(L):
            all_inspirations = selector.select(N)
            prompts = [task.prompts_with_scores([selector.responses[s] for s in inspirations], [selector.rewards[s] for s in inspirations]) for inspirations in all_inspirations]
            responses = self.pipeline.generate([prompt for prompt in prompts for _ in range(K)])
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            selector.update(all_inspirations, [responses[c * K:(c + 1) * K] for c in range(C)], rewards.view(C, K).tolist())

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/agent.py", "Agent configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
