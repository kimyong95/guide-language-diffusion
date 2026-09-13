import contextlib
import dataclasses
import os
from itertools import accumulate
import torch

# Pins each rank to its own physical GPU before CUDA ever initializes, so each thread sees the correct device.
if "LOCAL_RANK" in os.environ:
    assert not torch.cuda.is_initialized(), "Please import Accelerator after this file."
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[int(os.environ["LOCAL_RANK"])]

from transformers import AutoTokenizer, AutoModelForImageTextToText, DynamicCache
from transformers.cache_utils import LinearAttentionLayer
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper
from vllm.model_executor.layers.utils import apply_penalties
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from utils import func_cache


@dataclasses.dataclass
class GenerateOutput:
    tokens: list              # 2D list (N, Lg), ragged
    texts: list               # list (N) of str
    entropies: torch.Tensor   # (N,) mean sampled-token negative log probability, in nats

class VarlenCache(DynamicCache):
    """Owns packed K/V, batched linear state, and the active instance order.

    Initialize with {instance_id: prompt_tokens}. After each forward, call
    advance({instance_id: next_token}) with only surviving instances. Mapping
    order determines batch order. New instances cannot join during decoding.
    lengths includes the prepared queries; histories locates their previous K/V.
    """

    def __init__(self, prompts, device, config):
        super().__init__(config=config)
        self.config = config
        self.device = device
        self.prepare(prompts, {i: slice(0, 0) for i in prompts})

    @property
    def instance_ids(self):
        return list(self.lengths)

    @property
    def output_positions(self):
        return [query.stop - 1 for query in self.queries]

    def prepare(self, tokens, histories):
        """Builds a forward's packed inputs from its tokens and prior K/V slices."""
        packed = []
        self.lengths, self.queries = {}, []
        self.histories = [histories[i] for i in tokens]
        for instance_id, query in tokens.items():
            history = histories[instance_id]
            length = history.stop - history.start
            self.queries.append(slice(len(packed), len(packed) + len(query)))
            self.lengths[instance_id] = length + len(query)
            packed.extend(query)
        self.input_ids = torch.tensor([packed], device=self.device, dtype=torch.long)

    def model_inputs(self):
        """Returns packed decoder inputs and independent query/KV boundaries."""
        query_lengths = [query.stop - query.start for query in self.queries]
        key_lengths = list(self.lengths.values())
        cu_q = torch.tensor(list(accumulate(query_lengths, initial=0)), device=self.device, dtype=torch.int32)
        cu_k = torch.tensor(list(accumulate(key_lengths, initial=0)), device=self.device, dtype=torch.int32)
        return dict(input_ids=self.input_ids, past_key_values=self, use_cache=True,
                    cu_seq_lens_q=cu_q, cu_seq_lens_k=cu_k,
                    max_length_q=max(query_lengths), max_length_k=max(key_lengths))

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        """Appends each instance's new K/V beside that instance's own history."""
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(key_states, value_states)
        keys, values = [], []
        for history, query in zip(self.histories, self.queries):
            if history.stop > history.start:
                keys.append(layer.keys[..., history, :])
                values.append(layer.values[..., history, :])
            keys.append(key_states[..., query, :])
            values.append(value_states[..., query, :])
        layer.keys, layer.values = torch.cat(keys, dim=-2), torch.cat(values, dim=-2)
        layer.device, layer.dtype = layer.keys.device, layer.keys.dtype
        return layer.keys, layer.values

    def advance(self, tokens):
        """Keeps the supplied instance IDs and queues one next token for each."""
        histories = {}
        offset = 0
        for instance_id, length in self.lengths.items():
            histories[instance_id] = slice(offset, offset + length)
            offset += length
        if not tokens:
            self.layers = []
        elif list(tokens) != self.instance_ids:
            rows = {instance_id: row for row, instance_id in enumerate(self.lengths)}
            indices = torch.tensor([rows[i] for i in tokens], device=self.device, dtype=torch.long)
            for layer in self.layers:
                if isinstance(layer, LinearAttentionLayer):
                    layer.reorder_cache(indices)
                    layer.batch_size = len(tokens)
        # K/V removal is fused into the next update's concatenation of surviving histories.
        self.prepare({i: [token] for i, token in tokens.items()}, histories)


def packed_linear_forward(original_forward, layer_idx):
    """Adapts a Qwen3.5 linear mixer while retaining its original kernels."""
    def forward(hidden_states, cache_params=None, attention_mask=None, **kwargs):

        if not isinstance(cache_params, VarlenCache):
            return original_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)

        cache = cache_params

        # decode
        if cache.has_previous_state(layer_idx):
            batch = hidden_states.reshape(len(cache.lengths), 1, hidden_states.shape[-1])
            output = original_forward(batch, cache_params=cache)
            return output.reshape_as(hidden_states)

        # prefill
        else:
            outputs, conv_states, recurrent_states = [], [], []
            for query in cache.queries:
                local_cache = DynamicCache(config=cache.config)
                outputs.append(original_forward(hidden_states[:, query], cache_params=local_cache))
                state = local_cache.layers[layer_idx]
                conv_states.append(state.conv_states)
                recurrent_states.append(state.recurrent_states)
            cache.update_conv_state(torch.cat(conv_states, dim=0), layer_idx)
            cache.update_recurrent_state(torch.cat(recurrent_states, dim=0), layer_idx)
            return torch.cat(outputs, dim=1)

    return forward


class Pipeline:
    """Flash Attention 2 generation over packed, variable-length instances."""

    ATTN_IMPLEMENTATION = "flash_attention_2"
    LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(self, model_name, max_memory=None, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0):
        """Loads a frozen model; max_memory optionally controls device placement."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation=self.ATTN_IMPLEMENTATION,
            device_map="auto", max_memory=max_memory,
        ).eval().requires_grad_(False)
        self.device = self.model.device
        self.text_config = self.model.config.text_config
        self.layers = self.model.get_decoder().layers
        for index, layer in enumerate(self.layers):
            if getattr(layer, "block_type", None) == "linear_attention":
                layer.linear_attn.forward = packed_linear_forward(layer.linear_attn.forward, index)

        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = eos if isinstance(eos, list) else ([] if eos is None else [eos])
        self.do_sample = temperature > 0
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.logits_processors = LogitsProcessorList()
        if self.do_sample:
            if temperature != 1.0:
                self.logits_processors.append(TemperatureLogitsWarper(temperature))
            if top_k:
                self.logits_processors.append(TopKLogitsWarper(top_k))
            if top_p < 1.0:
                self.logits_processors.append(TopPLogitsWarper(top_p))
            if min_p > 0.0:
                self.logits_processors.append(MinPLogitsWarper(min_p))

    def apply_penalties(self, logits, prompts, generated):
        """Applies vLLM's repetition and presence penalties to each row in place.

        Args:
            logits: (N, V) float, contiguous
            prompts: 2D list (N, Lp), ragged, each row's prompt tokens
            generated: 2D list (N, Lg), ragged, the tokens each row has generated so far

        Returns:
            logits, penalized: a logit of a token in the prompt or generated is divided by
            repetition_penalty when positive and multiplied when negative; then presence_penalty is
            subtracted from the logit of every generated token.
        """
        if self.repetition_penalty == 1.0 and self.presence_penalty == 0.0:
            return logits
        presence = torch.tensor([self.presence_penalty], device=logits.device)
        repetition = torch.tensor([self.repetition_penalty], device=logits.device)
        for row, (prompt, output) in enumerate(zip(prompts, generated)):    # one row at a time, so no padding to a common length
            apply_penalties(
                logits[row:row + 1],
                torch.tensor([prompt], device=logits.device, dtype=torch.long),
                torch.tensor([output], device=logits.device, dtype=torch.long),
                presence, torch.zeros_like(presence), repetition,
            )
        return logits

    @func_cache()
    def texts_to_tokens(self, prompts, system_prompt=None, enable_thinking=False):
        """Returns ragged chat-template token IDs, cached per prompt.

        Args:
            prompts: list (N) of str
            system_prompt: str | None, the system turn's text; None omits the turn
            enable_thinking: bool, Qwen's chat-template switch

        Returns:
            2D list (N, Lp), ragged
        """
        system = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
        return [
            self.tokenizer(self.tokenizer.apply_chat_template(system + [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)).input_ids
            for prompt in prompts
        ]

    def construct_position_ids(self, token_lists):
        """Builds packed sequence and RoPE positions from complete token histories.

        Args:
            token_lists: 2D list (N, L), ragged, each instance's full token history

        Returns:
            Long tensor (4, 1, total_tokens) on the pipeline device: row 0 the packed index, rows 1-3
            each instance's own position, repeated for the three mRoPE axes.
        """
        lengths = [len(tokens) for tokens in token_lists]
        packed = torch.arange(sum(lengths), dtype=torch.long).unsqueeze(0)
        own = torch.cat([torch.arange(length, dtype=torch.long) for length in lengths]) if lengths else torch.empty(0, dtype=torch.long)
        return torch.cat((packed, own.expand(3, -1))).unsqueeze(1).to(self.device)

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

    @contextlib.contextmanager
    def inject_hidden(self, hidden, system_tokens):
        """Adds given states to the system prompt at every layer's token-mixer input.

        Layer l computes h + mixer(norm_l(h)), norm_l its input_layernorm with weight (1 + w_l);
        inside the block, at the system prompt's positions, the mixer reads
        norm_l(h + (1 + w_l) ⊙ hidden[l]) instead. The residual h, the MLP and every other
        position are untouched.

        Args:
            hidden: (H, P, D), one state per layer per system-prompt token, the instances' tokens
                concatenated in packed order, each row of rms at most 1; not checked here.
            system_tokens: list, the system prompt's token ids, located as a subsequence of each
                forward's input ids. A forward without them, such as a decode step, is left alone.

        Yields:
            None. Outside the block nothing is injected.
        """
        assert hidden.shape[0] == len(self.layers), f"{hidden.shape[0]} states for {len(self.layers)} layers"
        n_states, n_tokens = hidden.shape[1], len(system_tokens)
        self.inject_positions = None

        def set_positions(module, args):
            ids = args[0][0].tolist()
            positions = torch.zeros_like(args[0], dtype=torch.bool)
            for start in range(len(ids) - n_tokens + 1):
                if ids[start:start + n_tokens] == system_tokens:
                    positions[0, start:start + n_tokens] = True
            assert int(positions.sum()) in (0, n_states), f"{int(positions.sum())} system-prompt positions for {n_states} states"
            self.inject_positions = positions

        def inject(norm, states):
            def hook(module, args):
                positions = self.inject_positions
                if positions is None or not positions.any():
                    return
                hidden_states = args[0].clone()    # a copy, so the residual the caller saved before the norm keeps h
                positions = positions.to(hidden_states.device)
                scaled = (1 + norm.weight.float()) * states.to(hidden_states.device).float()
                hidden_states[positions] = hidden_states[positions] + scaled.to(hidden_states.dtype)
                return (hidden_states, *args[1:])
            return hook

        handles = [self.model.get_decoder().embed_tokens.register_forward_pre_hook(set_positions)]    # a multimodal wrapper calls the decoder with input_ids=None, but the embedding it built inputs_embeds with still sees the ids
        handles += [layer.input_layernorm.register_forward_pre_hook(inject(layer.input_layernorm, states)) for layer, states in zip(self.layers, hidden)]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self.inject_positions = None

    def tokens_to_texts(self, token_lists):
        """
        Args:
            token_lists: 2D list (N, L), ragged

        Returns:
            list (N) of str, special tokens stripped.
        """
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in token_lists]

    def predict_logits(self, tokens):
        """
        Args:
            tokens: list (L), as yielded by rotate when one is live.

        Returns:
            (L, V) logits.
        """
        input_ids = torch.tensor([tokens], device=self.device)      # (1, L)
        positions = self.construct_position_ids([tokens])
        hidden = self.model.get_decoder()(input_ids=input_ids, position_ids=positions, use_cache=False)
        return self.model.get_output_embeddings()(hidden.last_hidden_state)[0].float()


    def log_probs(self, prompt_tokens, input_tokens):
        """Log probability of the input tokens, the last input token is a target only, never fed, so the forward runs Lp + Lg - 1 positions.

        Args:
            prompt_tokens: list (Lp), the conditioning prefix, nonempty
            input_tokens: list (Lg), the tokens to score; empty is allowed and scores nothing

        Returns:
            (Lg,) float32, entry t being log p(input_tokens[t] | prompt_tokens, input_tokens[:t]).
        """
        if not input_tokens:
            return torch.empty(0, device=self.device)
        logits = self.predict_logits(prompt_tokens + input_tokens[:-1])[len(prompt_tokens) - 1:]
        targets = torch.tensor(input_tokens, device=logits.device)
        return logits.log_softmax(-1).gather(1, targets[:, None])[:, 0]

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=1024):
        """Generates packed continuations with independent per-instance cache state.

        Args:
            prompt_tokens: 2D list (N, Lp), ragged, each nonempty
            max_new_tokens: int, per-instance token limit, including terminating EOS

        Returns:
            GenerateOutput in instance order. Entropies are mean sampled-token NLLs under
            the filtered distribution, or the raw distribution for greedy decoding.
        """
        if max_new_tokens < 0 or any(not tokens for tokens in prompt_tokens):
            raise ValueError("Require nonempty prompts and a nonnegative token limit")
        generated = [[] for _ in prompt_tokens]
        nll = torch.zeros(len(prompt_tokens), device=self.device)
        cache = VarlenCache(dict(enumerate(prompt_tokens)), self.device, self.text_config)
        position_ids = self.construct_position_ids(prompt_tokens)
        step = 0
        while cache.lengths and step < max_new_tokens:
            instance_ids = cache.instance_ids
            hidden = self.model.get_decoder()(**cache.model_inputs(), position_ids=position_ids).last_hidden_state
            logits = self.model.get_output_embeddings()(hidden[0, cache.output_positions]).float()
            logits = self.apply_penalties(logits, [prompt_tokens[i] for i in instance_ids], [generated[i] for i in instance_ids])
            log_probs = self.logits_processors(cache.input_ids, logits).log_softmax(-1)
            tokens = torch.multinomial(log_probs.exp(), 1) if self.do_sample else log_probs.argmax(-1, keepdim=True)
            nll[instance_ids] -= log_probs.gather(1, tokens)[:, 0].to(self.device)
            sampled = tokens[:, 0].tolist()
            for index, token in zip(instance_ids, sampled):
                generated[index].append(token)
            keep = [row for row, token in enumerate(sampled) if token not in self.eos_token_ids]
            positions = cache.output_positions
            position_ids = position_ids[:, :, [positions[row] for row in keep]] + 1
            cache.advance({instance_ids[row]: sampled[row] for row in keep})
            step += 1

        counts = torch.tensor([max(len(tokens), 1) for tokens in generated], device=self.device)
        return GenerateOutput(tokens=generated, texts=self.tokens_to_texts(generated), entropies=nll / counts)
