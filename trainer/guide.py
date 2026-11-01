"""Mutates config.task.name's seed program toward a higher reward by guided sampling of config.model.name inside SGLang,
its guides picked by agent.py's RPUCG over config.C chains, no training.

Every one of config.L steps samples each chain once: RPUCG picks up to N = 3 of the chain's nodes as its inspirations,
best first, and config.K new responses are sampled independently under a weighted sum of log-probabilities over them,
p_i being the model under the task's prompts of inspiration i alone. Every guide thinks at config.model.reasoning_effort
when config.model.enable_thinking, scoring the response after its own prompt. A response's answer is what follows its
</think> when thinking, empty when its thinking never ends, and the whole response otherwise. With three inspirations,
every response is sampled under log p_2 + log p_1 - log p_3, the middle inspiration the pivot, pulled toward the best and
pushed from the worst; with fewer, as the seed alone, under log p_1, the best alone.

The C × K responses are sampled together
by the pipeline's group_generate, each as one group of requests, one per guide. Each token of a response is sampled from
the softmax of its weighted sum at the model's generation config's temperature, with no top-k, top-p or repetition
penalty, and given to all of the group's requests, so its guides advance and stop together; a response fills
config.model.max_total_tokens with its group's longest prompt at most.

Each response's reward r is the task's evaluate, config.task.concurrency at a time, and RPUCG is updated as in agent.py.
Each step logs every reward and response to wandb, -inf and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/guide.py --config config/guide.py
"""

import sys

import torch
import wandb
from absl import flags
from ml_collections import config_flags

from agent import RPUCG
from base import BaseTrainer

FLAGS = flags.FLAGS

N = 3


def guide_weights(N):
    """Each guide's weight: [1, 1, -1] for N = 3, the middle pivot plus the best minus the worst, and the best alone
    otherwise.

    Args:
        N: int, the number of guides picked, at most 3, best first

    Returns:
        list (N) of float, summing to 1
    """
    return [1.0, 1.0, -1.0] if N == 3 else [float(i == 0) for i in range(N)]


class Trainer(BaseTrainer):

    def run(self):
        """Runs config.L steps of RPUCG selection, guided sampling, evaluation and update from the seed program."""
        task, L, C, K = self.task, self.config.L, self.config.C, self.config.K
        rpucg = RPUCG(task.init_program, task.evaluate([task.init_program])[0], C)
        for l in range(L):
            all_inspirations = rpucg.select(N)
            all_prompts, all_weights = [], []
            for inspirations in all_inspirations:
                all_prompts += [task.prompts([rpucg.responses[s] for s in inspirations])] * K
                all_weights += [guide_weights(len(inspirations))] * K
            responses = self.pipeline.group_generate(all_prompts, all_weights)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            rpucg.update(all_inspirations, [responses[c * K:(c + 1) * K] for c in range(C)], rewards.view(C, K).tolist())

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
