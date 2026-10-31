"""Mutates config.task.name's seed program toward a higher reward by in-context sampling of config.model.name inside
SGLang, in config.C independent chains, no guidance and no training.

Each chain's buffer holds up to config.N responses with distinct rewards, best first, starting as the task's seed
program alone, taken as a response. Every one of config.L steps samples config.K new responses per chain
independently, each under the task's prompts_with_scores of the chain's buffer with their rewards, thinking at
config.model.reasoning_effort when config.model.enable_thinking; all C × K responses come from one generate call. A
response's answer is what follows its </think> when thinking, empty when its thinking never ends, and the whole response
otherwise.

Each response's reward r is the task's evaluate, config.task.concurrency at a time. A response joins its chain's buffer
unless its reward is -inf, as without an EVOLVE-BLOCK or when the evaluator rejects its code, or one the buffer already
holds; the buffer then keeps its N highest-reward responses. Each step logs every reward and response to wandb, -inf and
"" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/agent.py --config config/agent.py
"""

import math
import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/agent.py", "Agent configuration.")


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of in-context sampling, evaluation and buffer update in each chain from the seed program."""
        task, L, C, K, N = self.task, self.config.L, self.config.C, self.config.K, self.config.N
        init_reward = task.evaluate([task.init_program])[0]
        buffers = [{init_reward: task.init_program} for _ in range(C)]  # reward -> response, best first
        for l in range(L):
            prompts = [task.prompts_with_scores(list(buffer.values()), list(buffer)) for buffer in buffers]
            responses = self.pipeline.generate([prompt for prompt in prompts for _ in range(K)])
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            for i, (response, reward) in enumerate(zip(responses, rewards.tolist())):
                if math.isfinite(reward):
                    buffers[i // K].setdefault(reward, response)
            buffers = [dict(sorted(buffer.items(), reverse=True)[:N]) for buffer in buffers]

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
