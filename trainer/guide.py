"""Mutates config.task.name's SimpleTES seed program toward a higher reward by sampling config.model.name inside vLLM, each
response anchored on the best buffer program and guided by every buffer program's weight, no training.

The buffer starts as the task's seed program alone, taken as a response, so there is no initial generation. Every one
of config.L iterations samples config.N new responses independently over the buffer's G responses, sorted by reward,
each under Σ_k W_k log p_k with W_k = 1[k = 0] + w_k, anchored on the best response 0, its thinking included; p_k is the
model under the task's prompts of response k, SimpleTES's generation prompt in tasks.py's templates with its code as the
one reference, thinking at config.model.reasoning_effort when config.model.enable_thinking, scoring the response after
its own prompt. A response's answer is what follows its </think> when thinking, empty when its thinking never ends, and
the whole response otherwise. The weights push the best program from those below it in proportion to the reward gap
d_k = r_k - r_0, normalised by the gap sum D = Σ_k |d_k|, so they sum to -1, and the best fills its own empty side,
w_0 = 1; W thus sums to 1, 2 on the best, and ties get 0. All rewards tied gives w = 0, each response sampling from the
best alone.

All N responses are sampled together by one llm.generate call of N × G vLLM requests, request (i, k) holding prompt k
and response i as group i, so every response's G guides run in parallel. GuideLogitsProcessor, a vLLM batch-level
logits processor, sees every request's logits at every step; it combines each group's rows as their weighted logits,
sampling as the weighted log probabilities since the weights sum to 1, samples one token from its softmax at the
requests' temperature, the model's generation config's, with no top-k or top-p filter, and leaves that token the only
finite logit in each of the group's rows, so all its guides advance by the same token and stop together.

A group's requests must share every step from their first: vLLM's engine core runs in this process, so it steps only
once a call has added every request; max_num_seqs, N², fits every request as G ≤ N, and max_num_batched_tokens,
config.model.max_total_tokens, the N copies of every prompt together (asserted), so all of them prefill in the first
step. Every request gets vLLM's default budget, config.model.max_total_tokens minus its own prompt's tokens, so reaching
it ends one request before the rest of its group. apply raises when a group is split, as by that or by a preemption once
the KV cache runs out, or its outputs differ in length.

Each response's reward r is the task's evaluate, config.task.concurrency at a time; a response without an EVOLVE-BLOCK
or one whose code the evaluator rejects scores -inf and is dropped. The buffer then keeps the N highest-reward responses
among the new ones and itself, the new ones first on ties. Each iteration logs every reward and response to wandb, -inf
and "" included.

Usage: CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python trainer/guide.py --config config/guide.py
"""

import math
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
import wandb
from absl import flags
from ml_collections import config_flags
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")


def guide_weights(rewards):
    """Each guide's weight, the best program pushed from those below it by its reward gaps to them.

    Args:
        rewards: list (G) of float, in descending order

    Returns:
        list (G) of float, W_k = 1[k = 0] + w_k summing to 1: w_k = d_k / D for k > 0, with d_k = r_k - r_0 and
        D = Σ_k |d_k|, and w_0 = 1[D > 0]; so W_0 = 2, or 1 when all rewards tie
    """
    rewards = torch.tensor(rewards, dtype=torch.float64)
    gaps = rewards - rewards[0]
    gap_sum = -gaps.sum()
    weights = torch.nan_to_num(gaps / gap_sum)
    weights[0] = 1 + (gap_sum > 0).double()
    return weights.tolist()


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
        rows = torch.tensor([[row for row, *_ in members] for members in groups.values()], device=logits.device)
        weights = torch.tensor([[weight for _, weight, *_ in members] for members in groups.values()], device=logits.device)
        temperatures = torch.tensor([members[0][3] for members in groups.values()], device=logits.device)
        logits_guided = (weights[:, :, None] * logits[rows]).sum(1) / temperatures[:, None]
        tokens = torch.multinomial(logits_guided.softmax(-1), 1, generator=self.generator)
        logits[rows.flatten()] = float("-inf")
        logits[rows, tokens] = 0
        return logits


class Trainer(BaseTrainer):

    def setup_pipeline(self):
        super().setup_pipeline(
            max_num_seqs=self.config.N ** 2,
            max_num_batched_tokens=self.config.model.max_total_tokens,
            gpu_memory_utilization=0.85,
            logits_processors=[GuideLogitsProcessor],
        )

    def guided_generate(self, prompts, weights, N):
        """Samples N responses together, each independently under Σ_k W_k log p_k, p_k being the model under prompts[k],
        thinking at config.model.reasoning_effort when config.model.enable_thinking.

        All responses are one llm.generate call of N × G requests, request (i, k) holding prompts[k] and response i.

        Args:
            prompts: list (G) of str, the guides' prompts
            weights: list (G) of float, W_k for guide k, summing to 1
            N: int, number of responses

        Returns:
            list (N) of str, each response's answer: after its </think> when thinking, empty when its thinking never
            ends; the whole response otherwise
        """
        cfg = self.config.model
        prompt_tokens = self.pipeline.texts_to_tokens(prompts, enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort)
        assert N * sum(map(len, prompt_tokens)) < cfg.max_total_tokens, f"{N} copies of prompts of {sum(map(len, prompt_tokens))} tokens together exceed max_num_batched_tokens"
        G = len(prompts)
        sampling_extra_args = [{"guide_group": i, "guide_weight": weight, "guide_size": G} for i in range(N) for weight in weights]
        return self.pipeline.generate(prompt_tokens * N, sampling_extra_args, enable_thinking=cfg.enable_thinking).texts[::G]

    def run(self):
        """Runs config.L iterations of guided sampling, evaluation and buffer update from the seed program."""
        task, L, N = self.task, self.config.L, self.config.N
        buffer = [{"response": task.init_program, "reward": task.evaluate([task.init_program])[0]}]
        for l in range(L):
            prompts = task.prompts([data_i["response"] for data_i in buffer])
            weights = guide_weights([data_i["reward"] for data_i in buffer])
            responses = self.guided_generate(prompts, weights, N)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            data = [{"response": response, "reward": reward} for response, reward in zip(responses, rewards.tolist()) if math.isfinite(reward)]
            buffer = sorted(data + buffer, key=lambda data_i: data_i["reward"], reverse=True)[:N]

            objective_evaluations = (l + 1) * N
            extra = {"best": buffer[0]["reward"]}
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards, extra=extra)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
