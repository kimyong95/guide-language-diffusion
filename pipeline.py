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
from transformers.generation.logits_process import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from utils import func_cache


@dataclasses.dataclass(frozen=True)
class TokenSampler:

    # official recomended parameters
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0

    @property
    def do_sample(self):
        return self.temperature > 0

    def logits_processors(self):
        """Returns the LogitsProcessorList that filters a (N, V) logits row before sampling; empty for greedy."""
        processors = LogitsProcessorList()
        if self.do_sample:
            if self.temperature != 1.0:
                processors.append(TemperatureLogitsWarper(self.temperature))
            if self.top_k:
                processors.append(TopKLogitsWarper(self.top_k))
            if self.top_p < 1.0:
                processors.append(TopPLogitsWarper(self.top_p))
            if self.min_p > 0.0:
                processors.append(MinPLogitsWarper(self.min_p))
        return processors

    def __call__(self, logits):
        """Draws one next token per row from the filtered logits.

        Args:
            logits: (N, V) float, one row per instance

        Returns:
            tokens: (N,) long, sampled, or argmax when temperature is 0
            log_probs: (N,) float, each token's log probability under the distribution it was drawn
                from, which is the model's own only when no filter is active
        """
        log_probs = self.logits_processors()(None, logits).log_softmax(-1)
        tokens = torch.multinomial(log_probs.exp(), 1) if self.do_sample else log_probs.argmax(-1, keepdim=True)
        return tokens[:, 0], log_probs.gather(1, tokens)[:, 0]


RAW_SAMPLER = TokenSampler(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0)    # the model's own distribution, so a sample's log_probs is its sampling log probability


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
        """Returns packed decoder inputs, each instance's own positions, and independent query/KV boundaries."""
        query_lengths = [query.stop - query.start for query in self.queries]
        key_lengths = list(self.lengths.values())
        positions = [position for length, query_length in zip(key_lengths, query_lengths) for position in range(length - query_length, length)]
        cu_q = torch.tensor(list(accumulate(query_lengths, initial=0)), device=self.device, dtype=torch.int32)
        cu_k = torch.tensor(list(accumulate(key_lengths, initial=0)), device=self.device, dtype=torch.int32)
        return dict(input_ids=self.input_ids, position_ids=torch.tensor([positions], device=self.device, dtype=torch.long),
                    past_key_values=self, use_cache=True,
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
        # K/V removal is fused into the next update's concatenation of surviving histories.
        self.prepare({i: [token] for i, token in tokens.items()}, histories)


class Pipeline:
    """Generation over packed, variable-length instances: flash attention by default; sdpa, one instance at a time, when a layer needs its own mask."""

    ATTN_IMPLEMENTATION = "flash_attention_2"
    LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(self, model_name, max_memory=None, attn_implementation=ATTN_IMPLEMENTATION):
        """Loads a frozen model; max_memory optionally controls device placement."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation=attn_implementation,
            device_map="auto", max_memory=max_memory,
        ).eval().requires_grad_(False)
        self.device = self.model.device
        self.config = self.model.config
        self.layers = self.model.get_decoder().layers

        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = eos if isinstance(eos, list) else ([] if eos is None else [eos])

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

    @staticmethod
    def get_token_positions(input_tokens, target_tokens):
        """Returns the positions covered by every occurrence of target_tokens in input_tokens.

        Args:
            input_tokens: (1, L) long, a forward's packed input ids
            target_tokens: list, the token ids to locate as a contiguous subsequence

        Returns:
            (1, L) bool, True at every token of every match, all False when there is none
        """
        ids = input_tokens[0].tolist()
        positions = torch.zeros_like(input_tokens, dtype=torch.bool)
        for start in range(len(ids) - len(target_tokens) + 1):
            if ids[start:start + len(target_tokens)] == target_tokens:
                positions[0, start:start + len(target_tokens)] = True
        return positions

    @contextlib.contextmanager
    def system_positions(self, system_tokens):
        """Tracks where the system prompt sits in each forward's packed input.

        Args:
            system_tokens: list, the system prompt's token ids, located as a subsequence of each
                forward's input ids

        Yields:
            dict whose "positions" entry is refreshed at every forward to the (1, L) bool mask of
            the system prompt in that forward's input ids, all False when it is absent, as in a
            decode step.
        """
        state = {}

        def hook(module, args):
            state["positions"] = self.get_token_positions(args[0], system_tokens)

        handle = self.model.get_decoder().embed_tokens.register_forward_pre_hook(hook)
        try:
            yield state
        finally:
            handle.remove()


    @contextlib.contextmanager
    def replace_hidden(self, hidden, system_tokens):
        """Replaces the residual stream entering every layer at the system prompt's positions.

        Args:
            hidden: (H, P, D), one state per layer per system-prompt token, the instances' tokens
                concatenated in packed order; layer 0's is the token embedding
            system_tokens: list, the system prompt's token ids, located as a subsequence of each
                forward's input ids. A forward without them, such as a decode step, is left alone.

        Yields:
            None. Outside the block nothing is replaced.
        """
        def replace(states, state):
            def hook(module, args):
                positions = state["positions"]
                if not positions.any():
                    return
                hidden_states = args[0].clone()
                hidden_states[positions.to(hidden_states.device)] = states.to(hidden_states.device, hidden_states.dtype)
                return (hidden_states, *args[1:])
            return hook

        with self.system_positions(system_tokens) as state:
            handles = [layer.register_forward_pre_hook(replace(states, state)) for layer, states in zip(self.layers, hidden)]
            try:
                yield
            finally:
                for handle in handles:
                    handle.remove()

    @torch.no_grad()
    def record_hidden(self, prompt_tokens, system_tokens):
        """Records the residual stream entering every layer at the system prompt's positions.

        Args:
            prompt_tokens: list (Lp), one instance's tokens, containing system_tokens
            system_tokens: list, the system prompt's token ids

        Returns:
            (H, P, D) float32, what replace_hidden with these states would leave unchanged.
        """
        recorded = []

        def hook(module, args):
            recorded.append(args[0][state["positions"].to(args[0].device)].float())

        with self.system_positions(system_tokens) as state:
            handles = [layer.register_forward_pre_hook(hook) for layer in self.layers]
            try:
                self.model.get_decoder()(input_ids=torch.tensor([prompt_tokens], device=self.device), use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
        return torch.stack(recorded)

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
        hidden = self.model.get_decoder()(input_ids=input_ids, use_cache=False)
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
    def generate(self, prompt_tokens, max_new_tokens=1024, sampler=TokenSampler()):
        """Generates packed continuations with independent per-instance cache state.

        Args:
            prompt_tokens: 2D list (N, Lp), ragged, each nonempty
            max_new_tokens: int, per-instance token limit, including terminating EOS
            sampler: TokenSampler, draws each next token from the raw logits

        Returns:
            GenerateOutput in instance order. Entropies are mean sampled-token NLLs under
            the sampler's distribution, the model's own for RAW_SAMPLER.
        """

        generated = [[] for _ in prompt_tokens]
        nll = torch.zeros(len(prompt_tokens), device=self.device)
        cache = VarlenCache(dict(enumerate(prompt_tokens)), self.device, self.config)
        step = 0
        while cache.lengths and step < max_new_tokens:
            instance_ids = cache.instance_ids
            hidden = self.model.get_decoder()(**cache.model_inputs()).last_hidden_state
            logits = self.model.get_output_embeddings()(hidden[0, cache.output_positions]).float()
            tokens, log_probs = sampler(logits)
            nll[instance_ids] -= log_probs.to(self.device)
            sampled = tokens.tolist()
            for index, token in zip(instance_ids, sampled):
                generated[index].append(token)
            keep = [row for row, token in enumerate(sampled) if token not in self.eos_token_ids]
            cache.advance({instance_ids[row]: sampled[row] for row in keep})
            step += 1

        counts = torch.tensor([max(len(tokens), 1) for tokens in generated], device=self.device)
        return GenerateOutput(tokens=generated, texts=self.tokens_to_texts(generated), entropies=nll / counts)
