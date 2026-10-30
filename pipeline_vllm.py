import dataclasses
import torch

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from utils import func_cache


@dataclasses.dataclass
class GenerateOutput:
    tokens: list              # 2D list (N, Lg), ragged
    texts: list               # list (N) of str


class Pipeline:
    """Generation with vLLM, tensor parallel over every visible GPU."""

    def __init__(self, model_name, max_total_tokens, seed=0, **kwargs):
        """Loads the model's language part and its generation config; max_total_tokens bounds a prompt and its
        response together, kwargs go to vllm.LLM, such as max_num_seqs or logits_processors."""
        self.llm = LLM(
            model=model_name, tensor_parallel_size=torch.cuda.device_count(), max_model_len=max_total_tokens,
            generation_config="auto", seed=seed, language_model_only=True, **kwargs,
        )
        self.sampling_kwargs = {**self.llm.model_config.get_diff_sampling_param(), "max_tokens": None}
        self.tokenizer = self.llm.get_tokenizer()
        self.think_end = self.tokenizer.convert_tokens_to_ids("</think>")

    @func_cache()
    def texts_to_tokens(self, prompts, system_prompt=None, enable_thinking=False, reasoning_effort=None):
        """Returns ragged chat-template token IDs, cached per prompt.

        Args:
            prompts: list (N) of str
            system_prompt: str | None, the system turn's text; None omits the turn
            enable_thinking: bool, Qwen's chat-template switch; False raises for thinking-only templates
                that ignore it
            reasoning_effort: str | None, Qwen's chat-template thinking budget, e.g. "medium"; None leaves the
                template's default

        Returns:
            2D list (N, Lp), ragged
        """
        assert enable_thinking or "enable_thinking" in self.tokenizer.chat_template or "<think>" not in self.tokenizer.chat_template, "chat template does not support enable_thinking=False"
        system = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
        effort = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
        return [
            self.tokenizer(self.tokenizer.apply_chat_template(system + [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking, **effort)).input_ids
            for prompt in prompts
        ]

    def tokens_to_texts(self, token_lists, enable_thinking=False):
        """
        Args:
            token_lists: 2D list (N, L), ragged, responses
            enable_thinking: bool, as passed to texts_to_tokens

        Returns:
            list (N) of str, special tokens stripped: each response's answer, after its </think> when thinking, empty
            when its thinking never ends; the whole response otherwise
        """
        if enable_thinking:
            token_lists = [t[t.index(self.think_end) + 1:] if self.think_end in t else [] for t in token_lists]
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in token_lists]

    def generate(self, prompt_tokens, sampling_extra_args=None, enable_thinking=False):
        """Samples one response per prompt in one llm.generate call, under the model's generation config, each
        response filling max_total_tokens with its prompt at most.

        Args:
            prompt_tokens: 2D list (N, Lp), ragged, each nonempty
            sampling_extra_args: list (N) of dict, each request's SamplingParams.extra_args, as logits processors read
                them; None for none
            enable_thinking: bool, as passed to texts_to_tokens

        Returns:
            GenerateOutput in prompt order, a terminating EOS kept in the tokens, the texts from tokens_to_texts
        """
        sampling_extra_args = sampling_extra_args or [None] * len(prompt_tokens)
        request_outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=tokens) for tokens in prompt_tokens],
            [SamplingParams(**self.sampling_kwargs, extra_args=extra_args) for extra_args in sampling_extra_args],
            use_tqdm=False,
        )
        tokens = [list(request_output.outputs[0].token_ids) for request_output in request_outputs]
        return GenerateOutput(tokens=tokens, texts=self.tokens_to_texts(tokens, enable_thinking))
