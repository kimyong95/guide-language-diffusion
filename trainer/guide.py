"""Mutates config.task.name's seed program toward a higher reward by sampling config.model.name inside vLLM, each
response pivoted on the in-context distribution of the best responses so far and guided by their NFT weights, no
training.

The buffer holds up to config.G responses with distinct rewards, best first, starting as the task's seed program alone,
taken as a response, so there is no initial generation. Every one of config.L iterations samples config.N new responses
independently, each under log p̂ + Σ_i w_i log p_i over the buffer's G responses. The pivot p̂ is the model under the
task's prompts_with_scores of the top config.M ≤ config.G responses with their rewards; p_i is the model under the
task's prompts of response i alone. Both think at config.model.reasoning_effort when config.model.enable_thinking,
scoring the response after their own prompt. A response's answer is what follows its </think> when thinking, empty when
its thinking never ends, and the whole response otherwise. The weights are NFT's on-policy coefficients of the min-max
normalised rewards r̃_i = (r_i - r_min) / (r_max - r_min): w_i = (r̃_i - r̄) / (G (1 - r̄)), r̄ the mean of r̃, so they
sum to 0, are negative exactly below the mean and give the best 1/G. All rewards tied gives w = 0, each response
sampling from p̂ alone.

The N responses are sampled one after another, each by one llm.generate call of G + 1 vLLM requests as one group,
request k holding guide k's prompt, p̂'s first, so the response's G + 1 guides run in parallel. GuideLogitsProcessor, a
vLLM batch-level logits processor, sees every request's logits at every step; it combines each group's rows as their
weighted logits, sampling as the weighted log probabilities since the weights sum to 1, samples one token from its
softmax at the requests' temperature, the model's generation config's, with no top-k or top-p filter, and leaves that
token the only finite logit in each of the group's rows, so all its guides advance by the same token and stop together.

A group's requests must share every step from their first: vLLM's engine core runs in this process, so it steps only
once a call has added every request; max_num_seqs, G + 1, fits every request, and max_num_batched_tokens,
config.model.max_total_tokens, every prompt together (asserted), so all of them prefill in the first step. Every request gets vLLM's default budget, config.model.max_total_tokens minus its own prompt's tokens, so reaching
it ends one request before the rest of its group. apply raises when a group is split, as by that or by a preemption once
the KV cache runs out, or its outputs differ in length.

Each response's reward r is the task's evaluate, config.task.concurrency at a time. A response joins the buffer unless
its reward is -inf, as without an EVOLVE-BLOCK or when the evaluator rejects its code, or one the buffer already holds;
the buffer then keeps its G highest-reward responses. Each iteration logs every reward and response to wandb, -inf and
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
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/guide.py", "Guidance configuration.")


def guide_weights(rewards):
    """Each buffer program's weight, NFT's on-policy coefficient of its min-max normalised reward.

    Args:
        rewards: list (G) of float, finite

    Returns:
        list (G) of float, w_i = (r̃_i - r̄) / (G (1 - r̄)) summing to 0, with r̃_i = (r_i - r_min) / (r_max - r_min)
        and r̄ the mean of r̃; all 0 when all rewards tie
    """
    rewards = torch.tensor(rewards, dtype=torch.float64)
    rewards_normalised = torch.nan_to_num((rewards - rewards.min()) / (rewards.max() - rewards.min()))
    reward_mean = rewards_normalised.mean()
    return ((rewards_normalised - reward_mean) / (len(rewards) * (1 - reward_mean))).tolist()


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
            max_num_seqs=self.config.G + 1,
            max_num_batched_tokens=self.config.model.max_total_tokens,
            gpu_memory_utilization=0.85,
            logits_processors=[GuideLogitsProcessor],
        )

    def guided_generate(self, prompts, weights, N):
        """Samples N responses one after another, each independently under Σ_k W_k log p_k, p_k being the model under
        prompts[k], thinking at config.model.reasoning_effort when config.model.enable_thinking.

        Each response is one llm.generate call of K requests as one group, request k holding prompts[k].

        Args:
            prompts: list (K) of str, the guides' prompts
            weights: list (K) of float, W_k for guide k, summing to 1
            N: int, number of responses

        Returns:
            list (N) of str, each response's answer: after its </think> when thinking, empty when its thinking never
            ends; the whole response otherwise
        """
        cfg = self.config.model
        prompt_tokens = self.pipeline.texts_to_tokens(prompts, enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort)
        assert sum(map(len, prompt_tokens)) < cfg.max_total_tokens, f"prompts of {sum(map(len, prompt_tokens))} tokens together exceed max_num_batched_tokens"
        sampling_extra_args = [{"guide_group": 0, "guide_weight": weight, "guide_size": len(prompts)} for weight in weights]
        return [self.pipeline.generate(prompt_tokens, sampling_extra_args, enable_thinking=cfg.enable_thinking).texts[0] for _ in range(N)]

    def run(self):
        """Runs config.L iterations of guided sampling, evaluation and buffer update from the seed program."""
        task, L, N, G, M = self.task, self.config.L, self.config.N, self.config.G, self.config.M
        buffer = {task.evaluate([task.init_program])[0]: task.init_program}  # reward -> response, best first
        for l in range(L):
            rewards_buffer, responses_buffer = list(buffer), list(buffer.values())
            prompts = [task.prompts_with_scores(responses_buffer[:M], rewards_buffer[:M])] + task.prompts(responses_buffer)
            weights = [1.0] + guide_weights(rewards_buffer)
            responses = self.guided_generate(prompts, weights, N)
            rewards = torch.tensor(task.evaluate(responses), dtype=torch.float64)
            for response, reward in zip(responses, rewards.tolist()):
                if math.isfinite(reward):
                    buffer.setdefault(reward, response)
            buffer = dict(sorted(buffer.items(), reverse=True)[:G])

            objective_evaluations = (l + 1) * N
            self.log_rewards(objective_evaluations=objective_evaluations, rewards=rewards, extra={"best": max(buffer)})
            self.log_texts(objective_evaluations=objective_evaluations, rewards=rewards, texts=responses)

        wandb.finish()


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
