"""Runs simpletes-baseline.py's SimpleTES with config.model.name inside vLLM, each chain's in-context prompt replaced by
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

The C × K responses are sampled by config.K llm.generate calls one after another, call k sampling response k of every
chain together: chain c's G requests are group c, request j holding guide j's prompt, so every response's guides run in
parallel. GuideLogitsProcessor, a vLLM batch-level logits processor, sees every request's logits at every step; it
combines each group's rows as their weighted logits, sampling as the weighted log probabilities since the weights sum to
1, samples one token from its softmax at the requests' temperature, the model's generation config's, with no top-k or
top-p filter, and leaves that token the only finite logit in each of the group's rows, so all its guides advance by the
same token and stop together.

A group's requests must share every step from their first: vLLM's engine core runs in this process, so it steps only
once a call has added every request; max_num_seqs, C × N, fits every request of a call, and max_num_batched_tokens,
config.model.max_total_tokens, all of its prompts together (asserted), so all of them prefill in the first step. Every
request gets vLLM's default budget, config.model.max_total_tokens minus its own prompt's tokens, so reaching it ends one
request before the rest of its group. apply raises when a group is split, as by that or by a preemption once the KV
cache runs out, or its outputs differ in length.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/simpletes-guide.py --config config/simpletes-guide.py
"""

import importlib
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from absl import flags
from ml_collections import config_flags
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

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


class GuideLogitsProcessor(LogitsProcessor):
    """Samples each guide group's next token under Σ_k W_k log p_k at the group's temperature, the same token for all
    of the group's requests.

    A request joins group extra_args["guide_group"], of extra_args["guide_size"] requests, with weight
    W_k = extra_args["guide_weight"]; requests without "guide_group" are left untouched. vLLM applies temperature only
    after this processor, so it takes the requests' own.
    """

    def __init__(self, vllm_config, device, is_pin_memory):
        self.req_info = {}
        self.generator = torch.Generator(device=device).manual_seed(vllm_config.model_config.seed)

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(
            self.req_info,
            batch_update,
            lambda params, _, output_tok_ids: (params.extra_args["guide_group"], params.extra_args["guide_weight"], params.extra_args["guide_size"], params.temperature, output_tok_ids) if params.extra_args and "guide_group" in params.extra_args else None,
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Samples one token per group from its guided log probabilities at its temperature and leaves it the only finite
        logit in the group's rows.

        Args:
            logits: (B, V), one row per request in the batch

        Returns:
            (B, V), modified in place

        Raises:
            RuntimeError: a group has other than its guide_size requests in the batch, or their outputs differ in length
        """
        if not self.req_info:
            return logits
        groups = {}
        for row, (group, weight, size, temperature, output_tok_ids) in self.req_info.items():
            groups.setdefault(group, []).append((row, weight, size, temperature, len(output_tok_ids)))
        if any(len(members) != members[0][2] for members in groups.values()):
            raise RuntimeError(f"guide groups split across steps, as by a request reaching its length limit or a preemption, requests per group: { {group: len(members) for group, members in groups.items()} }")
        if any(len({length for *_, length in members}) > 1 for members in groups.values()):
            raise RuntimeError(f"guide groups out of sync, output lengths per group: { {group: [length for *_, length in members] for group, members in groups.items()} }")
        logits_guided = torch.stack([
            sum(weight * logits[row] for row, weight, *_ in members) / members[0][3]
            for members in groups.values()
        ])
        tokens = torch.multinomial(logits_guided.softmax(-1), 1, generator=self.generator)
        for members, token in zip(groups.values(), tokens):
            rows = [row for row, *_ in members]
            logits[rows] = float("-inf")
            logits[rows, token] = 0
        return logits


class Trainer(SimpletesTrainer):

    def setup_pipeline(self):
        super().setup_pipeline(
            max_num_seqs=self.config.C * self.config.N,
            max_num_batched_tokens=self.config.model.max_total_tokens,
            logits_processors=[GuideLogitsProcessor],
        )

    def generate(self, all_inspirations, all_failure_patterns):
        """Samples config.K responses per chain under log p_i* + Σ_j w_j log p_j, one guide prompt per inspiration,
        i* the best inspiration; Args and Returns as SimpletesTrainer.generate, each chain's prompt being i*'s."""
        all_prompts, all_weights, pivot_prompts = [], [], []
        for inspirations, failure_patterns in zip(all_inspirations, all_failure_patterns):
            prompts = [self.generator.build_prompt([node], failure_patterns) for node in inspirations]
            pivot = max(range(len(inspirations)), key=lambda j: inspirations[j].score)
            all_prompts.append(prompts)
            all_weights.append([float(j == pivot) + weight for j, weight in enumerate(guide_weights_a([node.score for node in inspirations])[pivot])])
            pivot_prompts.append(prompts[pivot])
        return pivot_prompts, self.guided_generate(all_prompts, all_weights)

    def guided_generate(self, all_prompts, all_weights):
        """Samples config.K responses per chain, each of chain c's under Σ_j W_j log p_j, p_j being the model under
        all_prompts[c][j], thinking at config.model.reasoning_effort when config.model.enable_thinking.

        Response k of every chain comes from llm.generate call k of config.K, chain c's requests holding all of
        all_prompts[c] as group c.

        Args:
            all_prompts: list (C) of list (G_c) of str, each chain's guides' prompts
            all_weights: list (C) of list (G_c) of float, each chain's W_j = 1[j = i*] + w_j, summing to 1

        Returns:
            list (C × K) of str, chain c's response k at c × K + k, each its answer: after its </think> when thinking,
            empty when its thinking never ends; the whole response otherwise
        """
        cfg, K = self.config.model, self.config.K
        prompt_tokens, sampling_extra_args, group_starts = [], [], []
        for c, (prompts, weights) in enumerate(zip(all_prompts, all_weights)):
            group_starts.append(len(prompt_tokens))
            prompt_tokens += self.pipeline.texts_to_tokens(prompts, enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort)
            sampling_extra_args += [{"guide_group": c, "guide_weight": weight, "guide_size": len(prompts)} for weight in weights]
        assert sum(map(len, prompt_tokens)) < cfg.max_total_tokens, f"prompts of {sum(map(len, prompt_tokens))} tokens together exceed max_num_batched_tokens"
        all_texts = [self.pipeline.generate(prompt_tokens, sampling_extra_args, enable_thinking=cfg.enable_thinking).texts for _ in range(K)]
        return [texts[group_start] for group_start in group_starts for texts in all_texts]


if __name__ == "__main__":
    config_flags.DEFINE_config_file("config", "config/simpletes-guide.py", "SimpleTES guidance configuration.")
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
