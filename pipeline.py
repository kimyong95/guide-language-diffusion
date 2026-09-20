import contextlib
import dataclasses
import os
from itertools import accumulate
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
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
    """Generation over packed, variable-length instances with flash attention."""

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
            enable_thinking: bool, Qwen's chat-template switch; True also opens the think block in the
                prompt so the model starts inside it instead of generating "<think>\n"

        Returns:
            2D list (N, Lp), ragged
        """
        system = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
        suffix = "<think>\n" if enable_thinking else ""
        return [
            self.tokenizer(self.tokenizer.apply_chat_template(system + [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + suffix).input_ids
            for prompt in prompts
        ]

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

    def record_latent(self, prompt_tokens, input_tokens):
        """Each layer's input at the input tokens, teacher forced, with the graph kept.

        Args:
            prompt_tokens: list (Lp), the conditioning prefix, nonempty
            input_tokens: list (Lg), the tokens whose states are recorded, nonempty

        Returns:
            (H, Lg, D) in the model's dtype, differentiable into whatever adapts the model; layer 0's is
            the token embedding.
        """
        input_ids = torch.tensor([prompt_tokens + input_tokens], device=self.device)
        outputs = self.model.get_decoder()(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        return torch.stack(outputs.hidden_states[:-1])[:, 0, len(prompt_tokens):]

    @contextlib.contextmanager
    def inject_latent(self, latent):
        """Replaces the residual stream entering every layer at each forward's pad tokens.

        Args:
            latent: (H, P, D), one state per layer per pad token, in input order; layer 0's is the
                token embedding. A forward holding pad tokens must hold exactly P of them.

        Yields:
            None. A forward without pad tokens, such as a decode step, is left alone.
        """
        state = {}

        def locate(module, args):
            state["positions"] = args[0] == self.tokenizer.pad_token_id    # (1, L) bool

        def replace(states):
            def hook(module, args):
                positions = state["positions"]
                if not positions.any():
                    return
                assert positions.sum() == len(states), f"{int(positions.sum())} pad tokens for {len(states)} latent states"
                hidden_states = args[0].clone()
                hidden_states[positions.to(hidden_states.device)] = states.to(hidden_states.device, hidden_states.dtype)
                return (hidden_states, *args[1:])
            return hook

        handles = [self.model.get_decoder().embed_tokens.register_forward_pre_hook(locate)]
        handles += [layer.register_forward_pre_hook(replace(states)) for layer, states in zip(self.layers, latent)]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=1024, generation_config=None, strip_eos=False):
        """Generates packed continuations with independent per-instance cache state.

        Args:
            prompt_tokens: 2D list (N, Lp), ragged, each nonempty
            max_new_tokens: int, per-instance token limit, including terminating EOS
            generation_config: GenerationConfig | None, whose sampling fields build the logits
                processors; None takes the model's own, as loaded from generation_config.json.
                Sampling is greedy unless do_sample is set. Processors reading the history, such as
                repetition_penalty, see the generated tokens only, never the prompt.
            strip_eos: bool, whether a terminating EOS is dropped from the returned tokens; the
                entropies still count it

        Returns:
            GenerateOutput in instance order. Entropies are mean sampled-token NLLs under the
            filtered distribution, the model's own when the config sets no filter.
        """
        generation_config = self.model.generation_config if generation_config is None else generation_config
        logits_processors = self.model._get_logits_processor(generation_config, device=self.device)

        generated_tokens = [[] for _ in prompt_tokens]
        nll = torch.zeros(len(prompt_tokens), device=self.device)
        cache = VarlenCache(dict(enumerate(prompt_tokens)), self.device, self.config)
        step = 0
        while cache.lengths and step < max_new_tokens:
            instance_ids = cache.instance_ids
            outputs = self.model.get_decoder()(**cache.model_inputs())
            logits = self.model.get_output_embeddings()(outputs.last_hidden_state[0, cache.output_positions]).float()
            # (N, step), unpadded: every live instance has generated step tokens
            log_probs = logits_processors(torch.tensor([generated_tokens[i] for i in instance_ids], device=self.device, dtype=torch.long), logits).log_softmax(-1)
            tokens = torch.multinomial(log_probs.exp(), 1) if generation_config.do_sample else log_probs.argmax(-1, keepdim=True)
            nll[instance_ids] -= log_probs.gather(1, tokens)[:, 0]
            sampled = tokens[:, 0].tolist()
            for index, token in zip(instance_ids, sampled):
                generated_tokens[index].append(token)
            keep = [row for row, token in enumerate(sampled) if token not in self.eos_token_ids]
            cache.advance({instance_ids[row]: sampled[row] for row in keep})
            step += 1

        counts = torch.tensor([max(len(tokens), 1) for tokens in generated_tokens], device=self.device)
        if strip_eos:
            generated_tokens = [tokens[:-1] if tokens and tokens[-1] in self.eos_token_ids else tokens for tokens in generated_tokens]
        return GenerateOutput(tokens=generated_tokens, texts=self.tokens_to_texts(generated_tokens), entropies=nll / counts)
