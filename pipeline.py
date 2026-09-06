import contextlib
import dataclasses
import os
from itertools import accumulate
import torch

# Pins each rank to its own physical GPU before CUDA ever initializes, so each thread sees the correct device.
if "LOCAL_RANK" in os.environ:
    assert not torch.cuda.is_initialized(), "Please import Accelerator after this file."
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[int(os.environ["LOCAL_RANK"])]

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from utils import func_cache


@dataclasses.dataclass
class GenerateOutput:
    tokens: list              # 2D list (N, Lg), ragged
    texts: list               # list (N) of str
    entropies: torch.Tensor   # (N,) mean sampled-token negative log probability, in nats


class VarlenCache(DynamicCache):
    """Owns packed inputs, request boundaries, and KV storage for ragged decoding."""

    def __init__(self, prompts, device):
        super().__init__()
        self.device = device
        self.active = list(range(len(prompts)))
        self.lengths = [len(tokens) for tokens in prompts]
        self.query_lengths = self.lengths.copy()
        self.input_ids = torch.tensor([[token for tokens in prompts for token in tokens]], device=device, dtype=torch.long)
        self.position_ids = torch.tensor([[pos for length in self.lengths for pos in range(length)]], device=device, dtype=torch.long)
        self.indices = torch.arange(sum(self.lengths), device=device)

    def model_inputs(self):
        """Returns decoder arguments for the pending prefill or decode step."""
        cu_q = torch.tensor(list(accumulate(self.query_lengths, initial=0)), device=self.device, dtype=torch.int32)
        cu_k = torch.tensor(list(accumulate(self.lengths, initial=0)), device=self.device, dtype=torch.int32)
        return dict(input_ids=self.input_ids, position_ids=self.position_ids, past_key_values=self, use_cache=True,
                    cu_seq_lens_q=cu_q, cu_seq_lens_k=cu_k,
                    max_length_q=max(self.query_lengths), max_length_k=max(self.lengths))

    def last_hidden(self, hidden):
        """Selects the final query state of each active request for sampling."""
        positions = torch.tensor(list(accumulate(self.query_lengths)), device=hidden.device) - 1
        return hidden[0, positions]

    def advance(self, tokens, keep):
        """Queues sampled tokens for surviving request indices in the current batch."""
        offsets = list(accumulate(self.lengths, initial=0))
        # Cache.update appends all new KVs first; gather each prefix beside its new KV.
        self.indices = torch.cat([
            torch.cat((torch.arange(offsets[i], offsets[i + 1], device=self.device),
                       torch.tensor([offsets[-1] + j], device=self.device)))
            for j, i in enumerate(keep)
        ])
        self.input_ids = torch.tensor([[tokens[i] for i in keep]], device=self.device)
        self.position_ids = torch.tensor([[self.lengths[i] for i in keep]], device=self.device)
        self.active = [self.active[i] for i in keep]
        self.lengths = [self.lengths[i] + 1 for i in keep]
        self.query_lengths = [1] * len(keep)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        layer = self.layers[layer_idx]
        layer.keys = keys = keys[:, :, self.indices.to(keys.device), :]
        layer.values = values = values[:, :, self.indices.to(values.device), :]
        return keys, values


class Pipeline:
    """Flash Attention 2 generation over packed, variable-length requests."""

    ATTN_IMPLEMENTATION = "flash_attention_2"

    def __init__(self, model_name, max_memory=None, temperature=1.0, top_p=1.0, top_k=0):
        """Loads a frozen model; max_memory optionally controls device placement."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation=self.ATTN_IMPLEMENTATION,
            device_map="auto", max_memory=max_memory,
        ).eval().requires_grad_(False)
        self.device = self.model.device
        self.config = self.model.config
        self.config.use_cache = False
        self.layers = self.model.get_decoder().layers
        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = eos if isinstance(eos, list) else ([] if eos is None else [eos])
        self.do_sample = temperature > 0
        self.logits_processors = LogitsProcessorList()
        if self.do_sample:
            if temperature != 1.0:
                self.logits_processors.append(TemperatureLogitsWarper(temperature))
            if top_k:
                self.logits_processors.append(TopKLogitsWarper(top_k))
            if top_p < 1.0:
                self.logits_processors.append(TopPLogitsWarper(top_p))

    @contextlib.contextmanager
    def rotate(self, R, prompt_tokens, spans=None):
        """Rotates marked hidden states before every decoder layer, including its residual.

        Args:
            R: (hidden_size, hidden_size), one matrix shared across layers and marked tokens.
            prompt_tokens: Ragged token IDs; selected tokens must have nonzero IDs.
            spans: One half-open token range per prompt; None selects each whole prompt.

        Yields:
            Marked prompts to use inside the context. Template and generated positions outside
            the selected spans receive no direct rotation. Final normalization is not hooked.

        Notes:
            h ← E                        # embed_tokens(|ids|), never rotated
            for l = 0 … L-1:
                h[M] ← h[M] Rᵀ           # the hook: in-place on the stream
                h ← h + Attn_l(LN₁(h))
                h ← h + MLP_l(LN₂(h))
            logits ← LN_f(h) W_uᵀ        # no rotation here

        """
        dim = self.config.hidden_size
        if R.shape != (dim, dim):
            raise ValueError(f"Expected rotation shaped {(dim, dim)}, got {tuple(R.shape)}")
        spans = spans if spans is not None else [(0, len(tokens)) for tokens in prompt_tokens]
        if len(spans) != len(prompt_tokens) or any(
            not 0 <= start <= end <= len(tokens) for tokens, (start, end) in zip(prompt_tokens, spans)
        ):
            raise ValueError("Require one valid half-open span per prompt")
        marked_prompts = [
            tokens[:start] + [-token for token in tokens[start:end]] + tokens[end:]
            for tokens, (start, end) in zip(prompt_tokens, spans)
        ]
        state = {}

        def unmark(module, args):
            state["mask"] = args[0] < 0
            return (args[0].abs(), *args[1:])

        def rotate_hidden(module, args):
            if not state["mask"].any():
                return
            hidden = args[0]
            mask = state["mask"].to(hidden.device)
            dtype = torch.promote_types(torch.promote_types(hidden.dtype, R.dtype), torch.float32)
            rotated = hidden[mask].to(dtype) @ R.to(device=hidden.device, dtype=dtype).T
            result = hidden.clone()
            result[mask] = rotated.to(hidden.dtype)
            return (result, *args[1:])

        handles = []
        try:
            handles.append(self.model.get_decoder().embed_tokens.register_forward_pre_hook(unmark))
            for layer in self.layers:
                handles.append(layer.register_forward_pre_hook(rotate_hidden))
            yield marked_prompts
        finally:
            for handle in handles:
                handle.remove()
            state.clear()

    @func_cache()
    def texts_to_tokens(self, prompts, system_prompt=None, enable_thinking=False):
        """Returns ragged chat-template token IDs, cached per prompt."""
        system = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return [
            self.tokenizer(self.tokenizer.apply_chat_template(system + [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)).input_ids
            for prompt in prompts
        ]

    @func_cache()
    @torch.no_grad()
    def texts_to_embedding(self, prompts, layer, system_prompt=None, enable_thinking=False):
        """
        Args:
            prompts: list (N) of str
            layer: int, the decoder layer whose input is read
            system_prompt: str | None
            enable_thinking: bool

        Returns:
            (N, D) float32, the hidden state at each prompt's last token, which under the causal mask is the
            only position that has read the whole prompt. Cached per prompt, so only the prompts this call
            has not seen reach a forward, and the batch is whatever the caller asks for at once.
        """
        prompt_tokens = self.texts_to_tokens(prompts, system_prompt=system_prompt, enable_thinking=enable_thinking)
        lengths = torch.tensor([len(tokens) for tokens in prompt_tokens], device=self.device)                    # (N,)
        input_ids = torch.full((len(prompt_tokens), int(lengths.max())), self.tokenizer.pad_token_id, device=self.device)
        for i, tokens in enumerate(prompt_tokens):
            input_ids[i, :len(tokens)] = torch.tensor(tokens, device=self.device)
        attention_mask = torch.arange(input_ids.shape[1], device=self.device) < lengths[:, None]                 # (N, Lp) padded on the right, so a real token keeps the position id it would have had alone
        state = {}

        def capture(module, args):
            state["hidden"] = args[0]                                                                            # (N, Lp, D)

        handle = self.layers[layer].register_forward_pre_hook(capture)
        try:
            self.model.get_decoder()(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)    # the decoder alone, so neither the (N, Lp, V) the lm head would build nor the cache nothing here reads
        finally:
            handle.remove()

        return state["hidden"][torch.arange(len(prompt_tokens), device=self.device), lengths - 1].float()

    def tokens_to_texts(self, token_lists):
        """
        Args:
            token_lists: 2D list (N, L), ragged

        Returns:
            list (N) of str, special tokens stripped.
        """
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in token_lists]

    # ===================== scoring (one sample, plain forward) =================
    def predict_logits(self, tokens):
        """
        Args:
            tokens: list (L), as yielded by rotate when one is live.

        Returns:
            (L, V) logits.
        """
        input_ids = torch.tensor([tokens], device=self.device)      # (1, L)
        return self.model(input_ids=input_ids, use_cache=False).logits[0]

    def log_probs(self, prompt_tokens, input_tokens):
        """Scores only continuation tokens, without retaining prompt vocabulary logits."""
        if not input_tokens:
            return torch.empty(0, device=self.device)
        input_ids = torch.tensor([prompt_tokens + input_tokens[:-1]], device=self.device)
        hidden = self.model.get_decoder()(input_ids=input_ids, use_cache=False).last_hidden_state[0, len(prompt_tokens) - 1:]
        logits = self.model.get_output_embeddings()(hidden).float()
        targets = torch.tensor(input_tokens, device=logits.device)
        return logits.log_softmax(-1).gather(1, targets[:, None])[:, 0]

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=1024):
        """Generates ragged continuations using packed FA2 attention and a fresh KV cache.

        Args:
            prompt_tokens: Ragged lists of nonempty prompt token IDs.
            max_new_tokens: Per-request token limit, including terminating EOS.

        Returns:
            GenerateOutput in request order. Entropies are mean sampled-token NLLs under
            the filtered distribution, or the raw distribution for greedy decoding.
        """
        if max_new_tokens < 0 or any(not tokens for tokens in prompt_tokens):
            raise ValueError("Require nonempty prompts and a nonnegative token limit")
        generated = [[] for _ in prompt_tokens]
        nll = torch.zeros(len(prompt_tokens), device=self.device)
        cache = VarlenCache(prompt_tokens, self.device)
        for step in range(max_new_tokens):
            if not cache.active:
                break
            hidden = self.model.get_decoder()(**cache.model_inputs()).last_hidden_state # will update cache in-place
            logits = self.model.get_output_embeddings()(cache.last_hidden(hidden)).float()
            log_probs = self.logits_processors(cache.input_ids, logits).log_softmax(-1)
            tokens = torch.multinomial(log_probs.exp(), 1) if self.do_sample else log_probs.argmax(-1, keepdim=True)
            nll[cache.active] -= log_probs.gather(1, tokens)[:, 0].to(self.device)
            sampled = tokens[:, 0].tolist()
            for index, token in zip(cache.active, sampled):
                generated[index].append(token)
            keep = [i for i, token in enumerate(sampled) if token not in self.eos_token_ids]
            if not keep or step + 1 == max_new_tokens:
                break

            cache.advance(sampled, keep)

        counts = torch.tensor([max(len(tokens), 1) for tokens in generated], device=self.device)
        return GenerateOutput(tokens=generated, texts=self.tokens_to_texts(generated), entropies=nll / counts)
