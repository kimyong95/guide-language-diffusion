"""Generation with SGLang, tensor parallel over every visible GPU, from text prompts under the model's chat template.

generate samples one response per prompt. group_generate samples one response per group of prompts, one request per
prompt, through extend_sglang: each token is drawn from Σ_k W_k log p_k over the group's requests and given to all of
them, so they advance and stop together.
"""

import os
import uuid

import torch
from sglang.srt.utils.common import get_device_sm

from extend_sglang import GroupEngine, GroupLogitProcessor

MODELS = {
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
}


def model_path(model_name):
    """The Hugging Face repo of a MODELS name in the visible GPU's best format: NVFP4 from Blackwell (sm100), FP8 from
    Ada and Hopper (sm89), INT4 below; any other name as is."""
    if model_name not in MODELS:
        return model_name
    sm = get_device_sm()
    return MODELS[model_name]["nvfp4" if sm >= 100 else "fp8" if sm >= 89 else "int4"]


class Pipeline:
    """Generation with SGLang, tensor parallel over every visible GPU."""

    def __init__(self, model_name, max_total_tokens, enable_thinking=False, reasoning_effort=None, seed=0, **kwargs):
        """Loads the model and its generation config's sampling parameters.

        Args:
            model_name: str, a MODELS name or a Hugging Face repo
            max_total_tokens: int, bound on a prompt and its response together
            enable_thinking: bool, Qwen's chat-template switch; False raises for thinking-only templates that ignore it
            reasoning_effort: str | None, Qwen's chat-template thinking budget, e.g. "medium"; None leaves the
                template's default
            seed: int
            **kwargs: passed to sglang.Engine, such as mem_fraction_static
        """
        self.max_total_tokens = max_total_tokens
        self.enable_thinking = enable_thinking
        self.chat_template_kwargs = {"enable_thinking": enable_thinking} | ({} if reasoning_effort is None else {"reasoning_effort": reasoning_effort})
        self.engine = GroupEngine(
            model_path=model_path(model_name),
            tp_size=torch.cuda.device_count(),
            context_length=max_total_tokens + 2,  # SGLang caps a prompt and its response together at context_length - 2
            random_seed=seed,
            enable_custom_logit_processor=True,
            disable_radix_cache=True,
            cuda_graph_max_bs_decode=256,
            disable_custom_all_reduce=os.environ.get("NCCL_P2P_DISABLE") == "1",
            **kwargs,
        )
        self.tokenizer = self.engine.tokenizer_manager.tokenizer
        self.sampling_params = self.engine.tokenizer_manager.model_config.get_default_sampling_params()
        assert enable_thinking or "enable_thinking" in self.tokenizer.chat_template or "<think>" not in self.tokenizer.chat_template, "chat template does not support enable_thinking=False"

    def apply_chat_template(self, prompts):
        """Each prompt's chat-template token IDs, the prompt as the user turn of its own chat.

        Args:
            prompts: list (N) of str

        Returns:
            2D list (N, Lp), ragged
        """
        return [
            self.tokenizer(self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs)).input_ids
            for prompt in prompts
        ]

    def strip_thinking(self, response):
        """The response's answer: after its </think> when thinking, empty when its thinking never ends; the whole
        response otherwise."""
        if not self.enable_thinking:
            return response
        _, think_end, answer = response.partition("</think>")
        return answer if think_end else ""

    def generate(self, prompts):
        """Samples one response per prompt under the model's generation config, each filling max_total_tokens with its
        prompt at most.

        Args:
            prompts: list (N) of str

        Returns:
            list (N) of str, each response after strip_thinking
        """
        prompt_tokens = self.apply_chat_template(prompts)
        outputs = self.engine.generate(
            input_ids=prompt_tokens,
            sampling_params=[{**self.sampling_params, "max_new_tokens": self.max_total_tokens - len(tokens)} for tokens in prompt_tokens],
        )
        return [self.strip_thinking(output["text"]) for output in outputs]

    def group_generate(self, all_prompts, all_weights):
        """Samples one response per group of prompts, each token from Σ_k W_k log p_k at the generation config's
        temperature, p_k the model under the group's prompt k, with no top-k, top-p or repetition penalty, each response
        filling max_total_tokens with the group's longest prompt at most.

        Args:
            all_prompts: list (B) of list (G_b) of str, each group's prompts
            all_weights: list (B) of list (G_b) of float, each group's W_k, summing to 1

        Returns:
            list (B) of str, each response after strip_thinking
        """
        input_ids, sampling_params, group_starts = [], [], []
        for prompts, weights in zip(all_prompts, all_weights):
            prompt_tokens = self.apply_chat_template(prompts)
            group_starts.append(len(input_ids))
            input_ids += prompt_tokens
            custom_params = {"group": uuid.uuid4().hex, "size": len(prompts)}
            max_new_tokens = self.max_total_tokens - max(map(len, prompt_tokens))
            sampling_params += [
                {**self.sampling_params, "repetition_penalty": 1.0, "max_new_tokens": max_new_tokens, "custom_params": {**custom_params, "weight": weight}}
                for weight in weights
            ]
        outputs = self.engine.generate(input_ids=input_ids, sampling_params=sampling_params, custom_logit_processor=GroupLogitProcessor.to_str())
        return [self.strip_thinking(outputs[group_start]["text"]) for group_start in group_starts]
