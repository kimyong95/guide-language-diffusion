"""Generation with SGLang, tensor parallel over every visible GPU, from text prompts under the model's chat template.

generate samples one response per prompt. group_generate samples one response per group of prompts, one request per
prompt, through extend_sglang: each token is drawn from the group's guided log probabilities over its requests and
given to all of them, so they advance and stop together. Both sample under the model's SAMPLING_PARAMS of its mode,
thinking or instruct. log_probs scores responses after prompts.

SGLang holds its KV cache only while generating, released through torch_memory_saver between generations, so that the
task's evaluation has that GPU memory.
"""

import contextlib
import os
import uuid

import torch
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.utils.common import get_device_sm

from extend_sglang import GroupEngine, GroupLogitProcessor

QUANT_MODELS = {
    "qwen-coder-next": {
        "int4": "inference-optimization/Qwen3-Coder-Next.w4a16",
        "fp8": "Qwen/Qwen3-Coder-Next-FP8",
        "nvfp4": "RedHatAI/Qwen3-Coder-Next-NVFP4",
    },
    "qwen-next-instruct": {
        "int4": "RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16",
        "fp8": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
        "nvfp4": "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
    },
    "qwen-next-thinking": {
        "int4": "RedHatAI/Qwen3-Next-80B-A3B-Thinking-quantized.w4a16",
        "fp8": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
        "nvfp4": "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4",
    },
    "qwen3.8-27b": {
        "int4": "RedHatAI/Qwen3.8-27B-INT4",
        "fp8": "Qwen/Qwen3.8-27B-FP8",
        "nvfp4": "nvidia/Qwen3.8-27B-NVFP4",
    },
    "qwen3.5-4b": {
        "int4": "Qwen/Qwen3.5-4B",
        "fp8": "Qwen/Qwen3.5-4B",
        "nvfp4": "Qwen/Qwen3.5-4B",
    },
}

# Qwen's recommended sampling parameters for general tasks, from each model's card on Hugging Face.
SAMPLING_PARAMS = {
    "qwen-coder-next": {
        "instruct": {"temperature": 1.0, "top_p": 0.95, "top_k": 40},
    },
    "qwen-next-instruct": {
        "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
    },
    "qwen-next-thinking": {
        "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0},
    },
    "qwen3.8-27b": {
        "thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0},
        "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
    },
    "qwen3.5-4b": {
        "thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
        "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
    },
}


def model_path(model_name):
    """The Hugging Face repo of a QUANT_MODELS name in the visible GPU's best format: NVFP4 from Blackwell (sm100), FP8
    from Ada and Hopper (sm89), INT4 below."""
    sm = get_device_sm()
    return QUANT_MODELS[model_name]["nvfp4" if sm >= 100 else "fp8" if sm >= 89 else "int4"]


@contextlib.contextmanager
def kv_cache(engine):
    """Holds SGLang's KV cache over the block, releasing it after.

    Args:
        engine: sglang.Engine
    """
    engine.resume_memory_occupation([GPU_MEMORY_TYPE_KV_CACHE])
    yield
    engine.release_memory_occupation([GPU_MEMORY_TYPE_KV_CACHE])


class Pipeline:
    """Generation with SGLang, tensor parallel over every visible GPU, its KV cache held only while generating."""

    def __init__(self, model_name, max_total_tokens, enable_thinking=False, reasoning_effort=None, seed=0, **kwargs):
        """Loads the model, sampling under its SAMPLING_PARAMS of the mode enable_thinking picks.

        Args:
            model_name: str, a QUANT_MODELS and SAMPLING_PARAMS name
            max_total_tokens: int, bound on a prompt and its response together
            enable_thinking: bool, Qwen's chat-template switch; False raises for thinking-only templates that ignore it
            reasoning_effort: str | None, Qwen's chat-template thinking budget, e.g. "medium"; None leaves the
                template's default
            seed: int
            **kwargs: passed to sglang.Engine, such as mem_fraction_static

        Raises:
            KeyError: SAMPLING_PARAMS has no such model, or not its mode
        """
        self.max_total_tokens = max_total_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.sampling_params = SAMPLING_PARAMS[model_name]["thinking" if enable_thinking else "instruct"]
        self.engine = GroupEngine(
            model_path=model_path(model_name),
            tp_size=torch.cuda.device_count(),
            context_length=max_total_tokens + 2,  # SGLang caps a prompt and its response together at context_length - 2
            random_seed=seed,
            enable_custom_logit_processor=True,
            disable_radix_cache=True,
            cuda_graph_max_bs_decode=256,
            disable_custom_all_reduce=os.environ.get("NCCL_P2P_DISABLE") == "1",
            enable_memory_saver=True,
            **kwargs,
        )
        self.engine.release_memory_occupation([GPU_MEMORY_TYPE_KV_CACHE])
        self.tokenizer = self.engine.tokenizer_manager.tokenizer
        assert enable_thinking or "enable_thinking" in self.tokenizer.chat_template or "<think>" not in self.tokenizer.chat_template, "chat template does not support enable_thinking=False"

    def apply_chat_template(self, prompts, enable_thinking=None, reasoning_effort=None, system_prompts=None):
        """Each prompt's chat-template token IDs, the prompt as the user turn of its own chat, after its system prompt
        as the system turn when given.

        Args:
            prompts: list (N) of str
            enable_thinking: bool | None, Qwen's chat-template switch; None leaves the template's default
            reasoning_effort: str | None, Qwen's chat-template thinking budget, e.g. "medium"; None leaves the
                template's default
            system_prompts: list (N) of str | None, each chat's system turn; None for no system turn

        Returns:
            2D list (N, Lp), ragged
        """
        chat_template_kwargs = {}
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            chat_template_kwargs["reasoning_effort"] = reasoning_effort
        all_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        if system_prompts is not None:
            all_messages = [[{"role": "system", "content": system_prompt}, *messages] for system_prompt, messages in zip(system_prompts, all_messages)]
        return [
            self.tokenizer(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs)).input_ids
            for messages in all_messages
        ]

    def strip_thinking(self, response):
        """The response's answer: after its </think> when thinking, empty when its thinking never ends; the whole
        response otherwise."""
        if not self.enable_thinking:
            return response
        _, think_end, answer = response.partition("</think>")
        return answer if think_end else ""

    def generate(self, prompts, system_prompts=None, sampling_params=None):
        """Samples one response per prompt under the pipeline's sampling_params, each filling max_total_tokens with its
        prompt at most.

        Args:
            prompts: list (N) of str
            system_prompts: list (N) of str | None, each chat's system turn; None for no system turn
            sampling_params: dict | None, SGLang sampling parameters overriding the pipeline's, e.g. {"temperature": 0}

        Returns:
            list (N) of str, each response after strip_thinking
        """
        prompt_tokens = self.apply_chat_template(prompts, enable_thinking=self.enable_thinking, reasoning_effort=self.reasoning_effort, system_prompts=system_prompts)
        with kv_cache(self.engine):
            outputs = self.engine.generate(
                input_ids=prompt_tokens,
                sampling_params=[{**self.sampling_params, **(sampling_params or {}), "max_new_tokens": self.max_total_tokens - len(tokens)} for tokens in prompt_tokens],
            )
        return [self.strip_thinking(output["text"]) for output in outputs]

    def log_probs(self, prompts, responses):
        """Each response's log probability under the model at temperature 1 after each prompt, the response being its
        tokens and the EOS right after the prompt's chat template.

        Args:
            prompts: list (N) of str
            responses: list (M) of str

        Returns:
            (N, M) float64, the sum of the response's token log probabilities after the prompt; (M) int64, each
            response's tokens with its EOS
        """
        prompt_tokens = self.apply_chat_template(prompts, enable_thinking=self.enable_thinking, reasoning_effort=self.reasoning_effort)
        response_tokens = [self.tokenizer(response, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id] for response in responses]
        with kv_cache(self.engine):
            outputs = self.engine.generate(
                input_ids=[tokens + response for tokens in prompt_tokens for response in response_tokens],
                sampling_params={"max_new_tokens": 0},
                return_logprob=True,
                logprob_start_len=[len(tokens) - 1 for tokens in prompt_tokens for _ in response_tokens],
            )
        # Each list starts at the prompt's last token, whose entry is None.
        log_P = [sum(logprob for logprob, _, _ in output["meta_info"]["input_token_logprobs"][1:]) for output in outputs]
        return torch.tensor(log_P, dtype=torch.float64).view(len(prompts), len(responses)), torch.tensor([len(response) for response in response_tokens])

    def group_generate(self, all_prompts, all_utilities):
        """Samples one response per group of prompts, each token from log p_B + Σ_{g_k > 0} g_k Δ_k +
        min(Σ_{g_k < 0} g_k Δ_k, 0) under the pipeline's sampling_params, each response filling max_total_tokens with
        the group's longest prompt at most.

        Over a group of G prompts, p_k is the model under prompt k after its repetition and presence penalties, the base
        log p_B = (1/G) Σ_k log p_k, Δ_k = log p_k - log p_B and g_k = W_k - 1/G, W being extend_sglang's weights of the
        prompts' utilities. The temperature, top-k, top-p and min-p apply to it as they do in generate, so a group of one
        prompt samples as generate does.

        Args:
            all_prompts: list (B) of list (G_b) of str, each group's prompts
            all_utilities: list (B) of list (G_b) of float, finite, each group's prompts' utilities

        Returns:
            list (B) of str, each response after strip_thinking
        """
        input_ids, sampling_params, group_starts = [], [], []
        for prompts, utilities in zip(all_prompts, all_utilities):
            prompt_tokens = self.apply_chat_template(prompts, enable_thinking=self.enable_thinking, reasoning_effort=self.reasoning_effort)
            group_starts.append(len(input_ids))
            input_ids += prompt_tokens
            custom_params = {"group": uuid.uuid4().hex, "size": len(prompts)}
            max_new_tokens = self.max_total_tokens - max(map(len, prompt_tokens))
            sampling_params += [
                {**self.sampling_params, "max_new_tokens": max_new_tokens, "custom_params": {**custom_params, "utility": utility}}
                for utility in utilities
            ]
        with kv_cache(self.engine):
            outputs = self.engine.generate(input_ids=input_ids, sampling_params=sampling_params, custom_logit_processor=GroupLogitProcessor.to_str())
        return [self.strip_thinking(outputs[group_start]["text"]) for group_start in group_starts]
