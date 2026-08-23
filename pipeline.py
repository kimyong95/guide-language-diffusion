import contextlib
import dataclasses

import torch
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, ContinuousBatchingConfig, GenerationConfig


def intervene_token_id(position):
    """
    Args:
        position: int, a slot's index into x flattened

    Returns:
        int, always negative so it cannot collide with a real token; -1 is left to the engine's own
        TMP_TOKEN_ID. Its own inverse, so the same call maps a marked id back to the x row it reads.
    """
    return -2 - position


@dataclasses.dataclass
class GenerateOutput:
    tokens: list              # 2D list (N, Lg), ragged
    texts: list               # list (N) of str
    entropies: torch.Tensor   # (N,) nats/token, from the log prob the engine reports for each sampled token


class Pipeline:
    """ragged tokens (+ optional intervention) -> paged generation, logits, scoring."""

    INTERVENE_TOKEN = "<|intervene_pad|>"    # marks a slot in the prompt; intervene relabels it per sample before any forward

    def __init__(self, model_name, max_memory=None, max_memory_percent=0.5, temperature=1.0):
        """
        Args:
            model_name: str HF repo id or local path.
            max_memory: {device: bytes} | None, the devices to shard across; None = every visible GPU.
                The paged cache lives on one device, so this must resolve to a single GPU.
            max_memory_percent: float, share of free memory the paged cache may take; the rest is left for
                the scoring forwards, which run on the same card.
            temperature: float, 0.0 = greedy. Fixed for the process: the engine binds it when it is built.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_memory = max_memory or {i: torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="paged|flash_attention_2", device_map="auto", max_memory=max_memory,).eval()
        self.model.requires_grad_(False)    # only an intervention x or an adapter ever trains, never the base weights
        self.device = self.model.device

        self.config = self.model.config
        self.layers = self.model.get_decoder().layers
        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = eos if isinstance(eos, list) else [eos]

        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.INTERVENE_TOKEN]})
        self.intervene_token_id = self.tokenizer.convert_tokens_to_ids(self.INTERVENE_TOKEN)

        self.generation_config = GenerationConfig(
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_k=0, top_p=1.0,      # off explicitly: Qwen3's own generation config would truncate to 20/0.95
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.cb_config = ContinuousBatchingConfig(
            use_cuda_graph=(False, True),
            allow_block_sharing=False,
            max_memory_percent=max_memory_percent,
            return_logprobs=True,
        )

    # ===================== the intervention =====================================
    @contextlib.contextmanager
    def intervene(self, x, prompt_tokens):
        """
        Args:
            x: (N, Lx, D); row i replaces every layer's input at the Lx INTERVENE_TOKEN slots in prompt i.
            prompt_tokens: 2D list (N, Lp), ragged; each holding Lx slots, where texts_to_tokens put them.

        Yields:
            2D list (N, Lp), the same prompts with every slot relabelled to the id of the x row it reads,
            handed out in reading order. Every forward inside the block must be fed these, not the originals.
        """
        N, Lx, _ = x.shape
        intervene_token_ids = iter([intervene_token_id(nl) for nl in range(N * Lx)])    # x's rows flattened, so sample i takes i * Lx .. i * Lx + Lx - 1
        intervene_prompt_tokens = [ [next(intervene_token_ids) if token == self.intervene_token_id else token for token in tokens] for tokens in prompt_tokens ]
        state = {}

        def compute_x_slots(module, args):
            token_ids = args[0]                                                # (1, NL)
            state["mask"] = (token_ids <= intervene_token_id(0))[..., None]    # (1, NL, 1)
            state["x"] = rearrange(x, "N Lx D -> (N Lx) D")[intervene_token_id(token_ids).clamp(min=0)]    # (1, NL, D)
            return token_ids.clamp(min=0)

        def write_x(layer, args):
            return torch.where(state["mask"], state["x"], args[0])

        handles = [self.model.get_decoder().embed_tokens.register_forward_pre_hook(compute_x_slots)]
        handles += [layer.register_forward_pre_hook(write_x) for layer in self.layers]

        try:
            yield intervene_prompt_tokens
        finally:
            for handle in handles:
                handle.remove()

    @contextlib.contextmanager
    def unpaged(self):
        """Swaps out the paged attention kernels, which need the engine's block tables, for ones a plain forward can run.

        Yields:
            None; the paged kernels are back on exit, which is what generate needs.
        """
        attn_implementation = self.config._attn_implementation
        self.model.set_attn_implementation(attn_implementation.removeprefix("paged|"))
        try:
            yield
        finally:
            self.model.set_attn_implementation(attn_implementation)

    # ===================== tokens =============================================
    def texts_to_tokens(self, prompts, system_prompt=None, enable_thinking=False, n_intervene=0):
        """
        Args:
            prompts: list (N) of str
            system_prompt: str | None, prepended as a system turn; None omits the turn entirely.
            enable_thinking: bool, Qwen3's chat-template switch.
            n_intervene: int, INTERVENE_TOKEN slots opening every prompt's user content; 0 omits them.

        Returns:
            2D list (N, Lp), ragged (no padding); intervene turns the slots into per-sample ids, and only a
            forward inside that context may be fed a prompt still holding them.
        """
        system = [{"role": "system", "content": system_prompt}] if system_prompt else []
        intervention = self.INTERVENE_TOKEN * n_intervene    # its own token, so it also keeps a prompt's leading whitespace off the header's newline
        return [
            self.tokenizer(self.tokenizer.apply_chat_template(system + [{"role": "user", "content": intervention + prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)).input_ids
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

    # ===================== scoring (one sample, plain forward) =================
    def predict_logits(self, tokens):
        """
        Args:
            tokens: list (L), as yielded by intervene when one is live.

        Returns:
            (L, V) logits.
        """
        input_ids = torch.tensor([tokens], device=self.device)      # (1, L)
        return self.model(input_ids=input_ids).logits[0]

    def log_probs(self, prompt_tokens, input_tokens):
        """Scores input_tokens teacher-forced after prompt_tokens, in one cache-free forward.

        Args:
            prompt_tokens: list (Lp)
            input_tokens: list (Lg)

        Returns:
            (Lg,) log P(input_tokens[l] | prompt_tokens, input_tokens[:l]) for l in Lg
        """
        Lp = len(prompt_tokens)
        logits = self.predict_logits(prompt_tokens + input_tokens)[Lp - 1:-1].float()    # (Lg, V): next-token logits over each input pos
        return logits.log_softmax(dim=-1).gather(1, torch.tensor(input_tokens, device=logits.device)[:, None])[:, 0]         # (Lg,)

    # ===================== generation (the paged engine) =======================
    @torch.no_grad()
    def generate(self, prompt_tokens, x=None, max_new_tokens=128):
        """
        Args:
            prompt_tokens: 2D list (N, Lp), ragged; with x None any INTERVENE_TOKEN slots stay as their own
                untrained embedding row rather than becoming an intervention.
            x: (N, Lx, D) | None, one intervention per sample, written at that sample's slots.
            max_new_tokens: int cap per sample.

        Returns:
            GenerateOutput, ragged; each sample ends at eos (included) or the cap.
        """
        with self.intervene(x, prompt_tokens) if x is not None else contextlib.nullcontext(prompt_tokens) as inputs:
            outputs = self.model.generate_batch(inputs=inputs, max_new_tokens=max_new_tokens, generation_config=self.generation_config, continuous_batching_config=self.cb_config, progress_bar=False, persistent_manager=True)

        assert len(outputs) == len(prompt_tokens), f"{len(prompt_tokens) - len(outputs)} requests never came back: the engine's thread died"

        # generate_batch returns its results already reordered into submission order
        generated_tokens = [output.generated_tokens for output in outputs.values()]
        entropies = torch.tensor([-sum(output.logprobs) / max(len(output.logprobs), 1) for output in outputs.values()], device=self.device)    # (N,) one sampled path, so an estimate rather than the full entropy
        return GenerateOutput(tokens=generated_tokens, texts=self.tokens_to_texts(generated_tokens), entropies=entropies)
