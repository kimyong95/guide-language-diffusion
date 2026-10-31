"""Runs simpletes-baseline.py's SimpleTES with config.model.name inside SGLang, each chain's in-context prompt replaced by
guided sampling; the task, evaluation, reflection, database and logging are the baseline's.

Every one of config.L steps samples each of config.C chains once. SimpleTES's balance policy picks up to config.N of the
chain's nodes as its G guides, the best of them, i*, being its pivot. Guide j's prompt is the generator's prompt holding
node j alone with its metrics and reflection and the chain's failure patterns, and p_j is the model under it, thinking
at config.model.reasoning_effort when config.model.enable_thinking. The chain's config.K responses each sample
log p_i* + Σ_j w_j log p_j under the model's generation config, w_j the guide's algo_mode "A" weights of anchor i*: as
the best, i* has only gaps d_j = r_j - r_i* ≤ 0, so w_j = d_j / D⁻ with D⁻ = Σ_j |d_j| pushes it from the worse guides,
summing to -1, and w_i* = 1 fills its empty positive side; ties get 0. Each response's weights 1[j = i*] + w_j thus sum
to 1; a chain of tied guides, as the seed alone, samples its pivot alone. The pivot's prompt is its children's
llm_input, so their reflection is over it.

The C × K responses are sampled together by the pipeline's group_generate, each as one group of its chain's G
requests, request j holding guide j's prompt. Each token of a response is sampled from the softmax of its
log p_i* + Σ_j w_j log p_j at the model's generation config's temperature, with no top-k, top-p or repetition penalty,
and given to all of the group's requests, so its guides advance and stop together; a response fills
config.model.max_total_tokens with its group's longest prompt at most.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/simpletes-guide.py --config config/simpletes-guide.py
"""

import importlib
import sys

import torch
from absl import flags
from ml_collections import config_flags

SimpletesTrainer = importlib.import_module("simpletes-baseline").Trainer

FLAGS = flags.FLAGS


def guide_weights_a(rewards):
    """Each anchor's weights over the buffer programs, its reward gaps to them normalised per side.

    Args:
        rewards: list (G) of float

    Returns:
        list (G) of list (G) of float, w[i][k] for anchor i and guide k: d_ik / D⁺_i for d_ik > 0 and d_ik / D⁻_i for
        d_ik < 0, with d_ik = r_k - r_i and D⁺_i, D⁻_i the sums of i's positive and absolute negative gaps;
        w[i][i] = 1[D⁺_i = 0] - 1[D⁻_i = 0]; 0 for ties
    """
    rewards = torch.tensor(rewards, dtype=torch.float64)
    gaps = rewards[None, :] - rewards[:, None]
    gaps_positive, gaps_negative = gaps.clamp(min=0), gaps.clamp(max=0)
    gap_sums_positive, gap_sums_negative = gaps_positive.sum(1), -gaps_negative.sum(1)
    weights = torch.nan_to_num(gaps_positive / gap_sums_positive[:, None]) + torch.nan_to_num(gaps_negative / gap_sums_negative[:, None])
    return (weights + torch.diag((gap_sums_positive == 0).double() - (gap_sums_negative == 0).double())).tolist()


class Trainer(SimpletesTrainer):

    def generate(self, all_inspirations, all_failure_patterns):
        """Samples config.K responses per chain under log p_i* + Σ_j w_j log p_j, one guide prompt per inspiration,
        i* the best inspiration; Args and Returns as SimpletesTrainer.generate, each chain's prompt being i*'s."""
        K = self.config.K
        all_prompts, all_weights, pivot_prompts = [], [], []
        for inspirations, failure_patterns in zip(all_inspirations, all_failure_patterns):
            prompts = [self.generator.build_prompt([node], failure_patterns) for node in inspirations]
            pivot = max(range(len(inspirations)), key=lambda j: inspirations[j].score)
            all_prompts.append(prompts)
            all_weights.append([float(j == pivot) + weight for j, weight in enumerate(guide_weights_a([node.score for node in inspirations])[pivot])])
            pivot_prompts.append(prompts[pivot])
        responses = self.pipeline.group_generate([prompts for prompts in all_prompts for _ in range(K)], [weights for weights in all_weights for _ in range(K)])
        return pivot_prompts, responses


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/simpletes-guide.py", "SimpleTES guidance configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
