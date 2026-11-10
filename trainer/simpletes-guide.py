"""Runs simpletes-baseline.py's SimpleTES, alternating its in-context sampling and guide.py's guided sampling over
config.C chains, no training.

Every one of config.L steps samples config.K responses per chain. A SimpleTES step samples as simpletes-baseline.py
does, and its prompt and finite rewards join the chain's PromptsRewards. Step l is a guide step of every chain when l is
odd and above config.M, sampling as guide.py does over the chain's PromptsRewards.get(config.M) prompts and their
utilities, p_i being the model under prompt i. Any other step is a SimpleTES step of every chain.

A guide step's responses are a batch of their chain with no parents, raising no visit count, their llm_input the prompt
of the highest utility; their evaluation, reflection and commit to the chain are a SimpleTES step's, and their rewards
join no PromptsRewards. Logging is as in simpletes-baseline.py.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/simpletes-guide.py --config config/simpletes-guide.py
"""

import importlib
import sys

import wandb
from absl import flags
from ml_collections import config_flags

from trainer.selector import PromptsRewards

SimpletesTrainer = importlib.import_module("trainer.simpletes-baseline").Trainer

FLAGS = flags.FLAGS


class Trainer(SimpletesTrainer):

    def run(self):
        """Runs config.L SimpleTES or guide steps from the seed program."""
        L, C, K, M, N = self.config.L, self.config.C, self.config.K, self.config.M, self.config.N
        self.add_init_program()
        all_prompts_rewards = [PromptsRewards(self.config.reward_agg_fn) for _ in range(C)]
        for l in range(L):
            if l % 2 == 1 and l > M:
                all_prompts, all_utilities = zip(*[prompts_rewards.get(M) for prompts_rewards in all_prompts_rewards])
                all_in_contexts, chain_idxs = [[] for _ in range(C)], list(range(C))
                gen_ids = [self.generator.next_gen_id() for _ in range(C)]
                for gen_id, chain_idx in zip(gen_ids, chain_idxs):
                    self.selector.register_batch(gen_id, chain_idx, [], K)
                prompts = [prompts[utilities.index(max(utilities))] for prompts, utilities in zip(all_prompts, all_utilities)]
                responses = self.pipeline.group_generate([prompts for prompts in all_prompts for _ in range(K)], [utilities for utilities in all_utilities for _ in range(K)])
                rewards = self.commit(all_in_contexts, chain_idxs, gen_ids, prompts, responses)
            else:
                selections = []
                for _ in range(C):
                    in_contexts, failure_patterns, chain_idx = self.selector.select(self.db, N, stats=None, eval_concurrency=0, backpressure_multiplier=0)
                    gen_id = self.generator.next_gen_id()
                    self.selector.register_batch(gen_id, chain_idx, [node.id for node in in_contexts], K)
                    selections.append((in_contexts, failure_patterns, chain_idx, gen_id))
                all_in_contexts, all_failure_patterns, chain_idxs, gen_ids = zip(*selections)
                prompts, responses = self.generate(all_in_contexts, all_failure_patterns)
                rewards = self.commit(all_in_contexts, chain_idxs, gen_ids, prompts, responses)
                for chain_idx, prompt, rewards_c in zip(chain_idxs, prompts, rewards.view(C, K).tolist()):
                    all_prompts_rewards[chain_idx].set(prompt, rewards_c)

            objective_evaluations = (l + 1) * C * K
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards)
            self.log_responses(objective_evaluations=objective_evaluations, rewards=rewards, responses=responses)

        wandb.finish()


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/simpletes-guide.py", "SimpleTES-guide configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
