import contextlib
import dataclasses
import os
import torch

# Pins each rank to its own physical GPU before CUDA ever initializes, so each thread sees the correct device.
if "LOCAL_RANK" in os.environ:
    assert not torch.cuda.is_initialized(), "Please import Accelerator after this file."
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[int(os.environ["LOCAL_RANK"])]

from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, ContinuousBatchingConfig, GenerationConfig
from utils import func_cache


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
    ATTN_IMPLEMENTATION = "flash_attention_2"    # what every forward runs on outside generate, which switches to the paged variant for as long as the engine holds the model

    def __init__(self, model_name, max_memory=None, max_memory_percent=0.8, temperature=1.0, top_p=1.0, top_k=0):
        """
        Args:
            model_name: str HF repo id or local path.
            max_memory: {device: bytes} | None, the devices to shard across; None = every visible GPU.
                The paged cache lives on one device, so this must resolve to a single GPU.
            max_memory_percent: float, share of free memory the paged cache may take; the rest is left for
                the scoring forwards, which run on the same card.
            temperature: float, 0.0 = greedy. Fixed for the process: the engine binds it when it is built.
            top_p: float, 1.0 keeps the whole distribution.
            top_k: int, 0 keeps the whole distribution. Bound with temperature, and like it fixed for the process.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_memory = max_memory or {i: torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation=self.ATTN_IMPLEMENTATION, device_map="auto", max_memory=max_memory,).eval()
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
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.cb_config = ContinuousBatchingConfig(
            use_cuda_graph=(False, True),
            allow_block_sharing=False,
            max_memory_percent=max_memory_percent,
            return_logprobs=True,
        )
        self.cb_manager = self.model.init_continuous_batching(generation_config=self.generation_config, continuous_batching_config=self.cb_config)
        self.cb_manager.warmup()    # the paged cache and its graphs are taken here, before any plain forward has churned the allocator
        self.model.set_attn_implementation(self.ATTN_IMPLEMENTATION)    # building the manager switched the model to the paged kernels, and only generate wants them

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
    def paged(self):
        """Hands the model to the engine: the paged attention kernels its block tables need, and a loop to run them.

        Yields:
            None; on exit the plain kernels are back, which is what a scoring forward needs, and the loop's
            thread is gone, which is what the interpreter needs to exit at all.
        """
        self.cb_manager.switch_to_paged_attn(self.model)
        self.cb_manager.start()
        try:
            yield
        finally:
            self.cb_manager.stop(keep_for_next_session=True) # this will restore the attn_implementation

    @func_cache()
    def texts_to_tokens(self, prompts, system_prompt=None, enable_thinking=False, n_intervene=0):
        """
        Args:
            prompts: list (N) of str
            system_prompt: str | None, prepended as a system turn; None omits the turn entirely.
            enable_thinking: bool, Qwen3's chat-template switch.
            n_intervene: int, INTERVENE_TOKEN slots appended to the prompt's own user turn, right after the
                prompt text and still inside the turn; 0 appends nothing.

        Returns:
            2D list (N, Lp), ragged (no padding); intervene turns the slots into per-sample ids, and only a
            forward inside that context may be fed a prompt still holding them. Cached per prompt, so a
            caller may ask for the same one every step rather than holding it.
        """
        system = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return [
            self.tokenizer(self.tokenizer.apply_chat_template(system + [{"role": "user", "content": prompt + self.INTERVENE_TOKEN * n_intervene}], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)).input_ids
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
            only position that has read the whole prompt. There is no n_intervene: an embedding describes the
            prompt alone, so it cannot move when x does. Cached per prompt, so only the prompts this call
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
    def generate(self, prompt_tokens, x=None, max_new_tokens=1024):
        """
        Args:
            prompt_tokens: 2D list (N, Lp)
            x: (N, Lx, D) | None

        Returns:
            GenerateOutput, ragged; each sample ends at eos (included) or the cap.
        """
        with self.intervene(x, prompt_tokens) if x is not None else contextlib.nullcontext(prompt_tokens) as inputs, self.paged():   # the hooks first, so nothing sits between the loop starting and its first request
            request_ids = self.cb_manager.add_requests(inputs=inputs, max_new_tokens=max_new_tokens)
            outputs = {}
            while len(outputs) < len(request_ids):
                result = self.cb_manager.get_result()
                outputs[result.request_id] = result

        outputs = [outputs[request_id] for request_id in request_ids]    # they come back in completion order; add_requests handed the ids out in submission order
        generated_tokens = [output.generated_tokens for output in outputs]
        entropies = torch.tensor([-sum(output.logprobs) / max(len(output.logprobs), 1) for output in outputs], device=self.device)    # (N,) one sampled path, so an estimate rather than the full entropy
        return GenerateOutput(tokens=generated_tokens, texts=self.tokens_to_texts(generated_tokens), entropies=entropies)
