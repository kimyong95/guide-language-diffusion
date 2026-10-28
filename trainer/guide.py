"""Mutates config.task's SimpleTES seed program toward a higher reward by sampling config.model.name inside vLLM, each
response anchored on one buffer program and guided by every buffer program's weight, no training.

The buffer starts as the task's seed program alone, so there is no initial generation. Every one of config.L iterations
samples config.N new responses over the buffer's G programs, response n anchored on program i = n mod G, under
log p_i + Σ_k w_ik log p_k, its thinking included; p_k is the model under SimpleTES's own generation prompt holding
program k's code alone with its metrics, thinking at config.model.reasoning_effort when config.model.enable_thinking, scoring
response n after its own prompt. A response's answer is what follows its </think> when thinking, empty when its thinking
never ends, and the whole response otherwise. The weights follow config.algo_mode. "A" pulls
program i toward the programs above it and pushes it from those below in proportion to the reward gap d_ik = r_k - r_i,
each side normalised by its gap sum, D⁺_i = Σ_{d_ik > 0} d_ik and D⁻_i = Σ_{d_ik < 0} |d_ik|, so the positive weights sum
to 1 and the negative to -1; i fills its own empty side, w_ii = 1 as the best and -1 as the worst, and ties get 0. "B"
gives every anchor the same weights w_k = (r̃_k - r̄) / (G r̄ (1 - r̄)), r̃ the rewards min-max normalised to [0, 1] and
r̄ their mean, so they sum to 0. Each response's weights 1[k = i] + w_ik thus sum to 1; all rewards tied gives w = 0,
each response sampling from its anchor alone.

All N responses are sampled together by one llm.generate call of N × G vLLM requests, request (n, k) holding prompt k and
response n as group n, so every response's G guides run in parallel. GuideLogitsProcessor, a vLLM batch-level logits processor, sees
every request's logits at every step; it combines each group's rows as their weighted logits, sampling as the weighted
log probabilities since the weights sum to 1, samples one token from its softmax at the requests' temperature, the
model's generation config's, with no top-k or top-p filter, and leaves that token the only finite logit in each of the
group's rows, so all its guides advance by the same token and stop together.

A group's requests must share every step from their first: vLLM's engine core runs in this process, so it steps only
once a call has added every request; max_num_seqs, N², fits every request as G ≤ N, and max_num_batched_tokens,
config.model.max_total_tokens, the N copies of every prompt together (asserted), so all of them prefill in the first
step. Every request gets vLLM's default budget, config.model.max_total_tokens minus its own prompt's tokens, so reaching
it ends one request before the rest of its group. apply raises when a group is split, as by that or by a preemption once
the KV cache runs out, or its outputs differ in length.

A response's code is the seed with its EVOLVE-BLOCK replaced by the answer's, saved to config.output_dir as
circle-{l}-{n}.py; its reward r is score_from_metrics of the task's metrics. A response without an EVOLVE-BLOCK or a code
the evaluator rejects scores -inf and is dropped. The buffer then keeps the N highest-reward codes, with their metrics,
among the new ones and itself, the new ones first on ties. Each iteration logs every reward and code to wandb, -inf and
"" included.

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
from simpletes.node import score_from_metrics
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")


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


def guide_weights_b(rewards):
    """Every anchor's same weights over the buffer programs, its normalised rewards centred on their mean.

    Args:
        rewards: list (G) of float

    Returns:
        list (G) of list (G) of float, w[i][k] = (r̃_k - r̄) / (G r̄ (1 - r̄)) for every anchor i, r̃ the rewards
        min-max normalised to [0, 1] and r̄ their mean; 0 when all rewards tie
    """
    rewards = torch.tensor(rewards, dtype=torch.float64)
    G = len(rewards)
    if rewards.min() == rewards.max():
        return torch.zeros(G, G, dtype=torch.float64).tolist()
    rewards_normalized = (rewards - rewards.min()) / (rewards.max() - rewards.min())
    reward_mean = rewards_normalized.mean()
    return ((rewards_normalized - reward_mean) / (G * reward_mean * (1 - reward_mean))).expand(G, G).tolist()


GUIDE_WEIGHTS = {"A": guide_weights_a, "B": guide_weights_b}


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

    def __init__(self, config):
        super().__init__(config)
        os.makedirs(config.output_dir, exist_ok=True)

    def setup_pipeline(self):
        super().setup_pipeline(
            max_num_seqs=self.config.N ** 2,
            max_num_batched_tokens=self.config.model.max_total_tokens,
            logits_processors=[GuideLogitsProcessor],
        )

    def guided_generate(self, prompts, weights, N):
        """Samples N responses together, response n under log p_i + Σ_k w_ik log p_k with i = n mod G, p_k being the
        model under prompts[k], thinking at config.model.reasoning_effort when config.model.enable_thinking.

        All responses are one llm.generate call of N × G requests, request (n, k) holding prompts[k] and response n.

        Args:
            prompts: list (G) of str, the guides' prompts
            weights: list (G) of list (G) of float, w_ik for anchor i and guide k
            N: int, number of responses

        Returns:
            list (N) of str, each response's answer: after its </think> when thinking, empty when its thinking never
            ends; the whole response otherwise
        """
        cfg = self.config.model
        prompt_tokens = self.pipeline.texts_to_tokens(prompts, enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort)
        assert N * sum(map(len, prompt_tokens)) < cfg.max_total_tokens, f"{N} copies of prompts of {sum(map(len, prompt_tokens))} tokens together exceed max_num_batched_tokens"
        G = len(prompts)
        sampling_extra_args = [{"guide_group": n, "guide_weight": float(k == n % G) + weight, "guide_size": G} for n in range(N) for k, weight in enumerate(weights[n % G])]
        return self.pipeline.generate(prompt_tokens * N, sampling_extra_args, enable_thinking=cfg.enable_thinking).texts[::G]

    def run(self):
        """Runs config.L iterations of guided sampling, evaluation and buffer update from the seed program."""
        task, L, N = self.task, self.config.L, self.config.N
        init_metrics = task.evaluate([task.init_program])[0]
        init_metrics.pop("code")
        buffer = [{"code": task.init_program, "metrics": init_metrics, "reward": score_from_metrics(init_metrics)}]
        for l in range(L):
            prompts = [task.prompt([data_i["code"]], [data_i["metrics"]]) for data_i in buffer]
            weights = GUIDE_WEIGHTS[self.config.algo_mode]([data_i["reward"] for data_i in buffer])
            responses = self.guided_generate(prompts, weights, N)
            metrics = task.evaluate(responses)
            codes = [metrics_i.pop("code") for metrics_i in metrics]
            rewards = torch.tensor([score_from_metrics(metrics_i) for metrics_i in metrics], dtype=torch.float64)
            data = []
            for n, (code, metrics_i, reward) in enumerate(zip(codes, metrics, rewards.tolist())):
                with open(os.path.join(self.config.output_dir, f"circle-{l}-{n}.py"), "w") as f:
                    f.write(code or "")
                if math.isfinite(reward):
                    data.append({"code": code, "metrics": metrics_i, "reward": reward})
            buffer = sorted(data + buffer, key=lambda data_i: data_i["reward"], reverse=True)[:N]

            objective_evaluations = (l + 1) * N
            extra = {"best": buffer[0]["reward"]}
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards, extra=extra)
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=[code or "" for code in codes])

        wandb.finish()


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
