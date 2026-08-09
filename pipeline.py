"""Varlen KV-cache pipeline over Qwen3, with an optional hidden-state intervention.

A custom attention kernel registered via `AttentionInterface.register` keeps one ragged KV cache per
sample and attends the whole batch in a single `flash_attn_varlen_func` call -- no padding. Qwen3's
own forward stack runs untouched around it, with our varlen state riding in as a `varlen_kwargs={...}`
kwarg; `use_cache=False` because we own the cache, and a custom attn-impl name builds no attention
mask, so the kernel owns causality and sample isolation via cu_seqlens + causal=True.

`texts_to_tokens(prompts, n_intervene=Lx)` appends `<|vision_start|><|image_pad|>*Lx<|vision_end|>`
to the user content, and `intervene` overwrites those slots' hidden state with a per-sample
x (N, Lx, D) -- the same x at all H decoder layers' input. They stay ordinary prompt positions with
ordinary RoPE, so x is written once at prefill and its k/v then serves every decode step. `x=None` is
a plain forward, so one Pipeline scores and generates both with and without an intervention.

Shape symbols -- L* is a length in tokens, N* a number of samples: H = layers, L = a sample's token
length (L_i sample i's, Lk_i its cached length), NL = the packed axis (sum_i L_i over N samples),
Lp = prompt length, Lg = generated length, Lx = intervention slots, N = samples. Model dims:
heads_q/heads_kv = attn/kv heads, Dh = head dim, D = hidden size, V = vocab.
"""

import contextlib
import itertools

import einops
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, AttentionInterface
from flash_attn import flash_attn_varlen_func


# ===================== our attention kernels (the only custom compute) =======
def assemble_kv(kv_caches, key, value, cu_q, li):
    """
    Args:
        kv_caches: list (N) of DynamicCache | None, updated in place.
        key: (1, heads_kv, NL, Dh)
        value: (1, heads_kv, NL, Dh)
        cu_q: (N+1,) int32
        li: int layer index.

    Returns:
        (k, v), each (sum Lk_i, heads_kv, Dh) packed in sample order.
    """
    if kv_caches is None:
        return key[0].transpose(0, 1), value[0].transpose(0, 1)    # Lk_i == L_i, already packed
    k_full, v_full = [], []
    for i, kv in enumerate(kv_caches):                             # per sample (varlen: batch is a python list)
        ki, vi = kv.update(key[:, :, cu_q[i]:cu_q[i + 1]], value[:, :, cu_q[i]:cu_q[i + 1]], li)
        k_full.append(ki[0].transpose(0, 1))                       # (Lk_i, heads_kv, Dh)
        v_full.append(vi[0].transpose(0, 1))
    return torch.cat(k_full), torch.cat(v_full)


def varlen_attention(module, query, key, value, attention_mask, varlen_kwargs, **kwargs):
    """Called by Qwen3Attention through the AttentionInterface registry, never by us directly.

    Args:
        module: Qwen3Attention
        query: (1, heads_q, NL, Dh)
        key: (1, heads_kv, NL, Dh)
        value: (1, heads_kv, NL, Dh)
        attention_mask: Always None.
        varlen_kwargs: {"kv_caches": list (N) of DynamicCache | None, "cu_q": (N+1,) int32,
            "cu_k": (N+1,) int32}, bundled by Pipeline.predict_logits.

    Returns:
        ((NL, heads_q, Dh), None) -- the (attn_output, attn_weights) pair Qwen3Attention expects.
    """
    kv_caches, cu_q, cu_k = varlen_kwargs["kv_caches"], varlen_kwargs["cu_q"], varlen_kwargs["cu_k"]
    k, v = assemble_kv(kv_caches, key, value, cu_q, module.layer_idx)   # (sum Lk_i, heads_kv, Dh)
    max_q = (cu_q[1:] - cu_q[:-1]).max().item()
    max_k = (cu_k[1:] - cu_k[:-1]).max().item()
    out = flash_attn_varlen_func(
        query[0].transpose(0, 1).contiguous(),                     # (NL, heads_q, Dh)
        k, v,
        cu_q, cu_k, max_q, max_k,
        softmax_scale=module.scaling, causal=True)                 # (NL, heads_q, Dh)
    return out, None       # Qwen3Attention.forward reshapes (NL, heads_q, Dh) -> (1, NL, heads_q*Dh)


AttentionInterface.register("varlen_attention", varlen_attention)


# ===================== the loop's view: pack ragged tokens, get logits =======
class Pipeline:
    """ragged tokens + per-sample ragged caches (+ optional intervention) -> logits, generation, scoring (varlen)."""

    INTERVENE_TOKEN = "<|image_pad|>"     # the prompt slot whose hidden state the caller optimizes

    def __init__(self, model_id, max_memory=None):
        """
        Args:
            model_id: str HF repo id or local path.
            max_memory: {device: bytes} | None, the devices to shard across; None = every visible GPU.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        max_memory = max_memory or {i: torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="varlen_attention", device_map="auto", max_memory=max_memory,).eval()
        self.model.requires_grad_(False)                 # only an intervention x or an adapter ever trains, never the base weights
        self.device = self.model.device
        self.intervene_token_id = self.tokenizer.convert_tokens_to_ids(self.INTERVENE_TOKEN)

        # Read off the bare model now: a caller may replace self.model with a PEFT/DDP wrapper, and
        # neither forwards these through. The decoder layers survive wrapping as the same objects.
        self.config = self.model.config
        self.layers = self.model.get_decoder().layers
        self.eos_token_ids = self.model.generation_config.eos_token_id

    @contextlib.contextmanager
    def intervene(self, x, positions):
        """Overwrites the hidden state at `positions` with x at every decoder layer's input.

        One pre-hook per layer re-writes the same x at all H layers, so the slots never carry a
        layer's own output forward; Qwen3 computes each layer's k/v (with RoPE) from x as it would
        for any token. The caller locates the slots -- this knows nothing about tokenization,
        packing or the kernel.

        Args:
            x: (N, Lx, D) | None; row i replaces every layer's input at sample i's Lx slots. None
                registers no hooks.
            positions: (N*Lx,) int64 indices along the sequence axis, in sample order.

        Yields:
            None; the hooks are live inside the block and removed on exit. Differentiable in x.
        """
        if x is None:
            yield
            return
        assert positions.numel() == x.shape[0] * x.shape[1], f"{positions.numel()} slots for x (N={x.shape[0]}, Lx={x.shape[1]})"

        def pre_hook(layer, args):
            h = args[0]                                              # (N, L, D) residual stream -- (1, NL, D) when packed
            src = einops.repeat(x, "n lx d -> b (n lx) d", b=h.shape[0])
            return (h.index_copy(1, positions.to(h.device), src.to(h)),)  # new tensor: no in-place on the graph

        handles = [layer.register_forward_pre_hook(pre_hook) for layer in self.layers]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def texts_to_tokens(self, prompts, system_prompt=None, enable_thinking=False, n_intervene=0):
        """
        Args:
            prompts: list (N) of str
            system_prompt: str | None, prepended as a system turn; None omits the turn entirely.
            enable_thinking: bool, Qwen3's chat-template switch.
            n_intervene: int, INTERVENE_TOKEN slots appended to every prompt's user content.

        Returns:
            2D list (N, L), ragged (no padding)
        """
        intervention = f"<|vision_start|>{self.INTERVENE_TOKEN * n_intervene}<|vision_end|>" if n_intervene else ""
        system = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return [
            self.tokenizer(self.tokenizer.apply_chat_template(system + [{"role": "user", "content": prompt + intervention}], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)).input_ids
            for prompt in prompts
        ]

    def predict_logits(self, kv_caches, input_tokens, x=None):
        """
        Args:
            kv_caches: list (N) of DynamicCache | None, updated in place.
            input_tokens: 2D list (N, L), ragged; appended per sample.
            x: (N, Lx, D) | None, row i written to sample i's INTERVENE_TOKEN positions.

        Returns:
            (N, L, V) logits for each input_tokens.
        """
        dev = self.device
        past = [0] * len(input_tokens) if kv_caches is None else [kv.get_seq_length() for kv in kv_caches]  # per-sample length BEFORE append
        q_lens = [len(t) for t in input_tokens]
        input_ids = torch.tensor([list(itertools.chain.from_iterable(input_tokens))], device=dev)
        position_ids = torch.tensor([[p for pa, q in zip(past, q_lens) for p in range(pa, pa + q)]], device=dev)  # (1, NL)
        cu_q = torch.tensor(list(itertools.accumulate(q_lens, initial=0)), dtype=torch.int32, device=dev)  # query offsets
        cu_k = torch.tensor(list(itertools.accumulate((pa + q for pa, q in zip(past, q_lens)), initial=0)), dtype=torch.int32, device=dev)

        # official black-box forward; our varlen state rides in as a kwarg, cache stays ours (use_cache=False)
        varlen_kwargs = dict(kv_caches=kv_caches, cu_q=cu_q, cu_k=cu_k)
        slots = (input_ids[0] == self.intervene_token_id).nonzero()[:, 0]   # (N*Lx,) into the packed axis
        with self.intervene(x, slots):                      # no-op when x is None
            out = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False, varlen_kwargs=varlen_kwargs)
        return list(out.logits[0].split(q_lens))            # [ (L_i, V) ] per sample

    def log_probs(self, prompt_tokens, input_tokens, x=None):
        """Scores input_tokens teacher-forced after prompt_tokens, in one cache-free forward.

        Args:
            prompt_tokens: list (Lp)
            input_tokens: list (Lg)
            x: (1, Lx, D) | None -- one sample is scored, so N = 1

        Returns:
            (Lg,) log P(input_tokens[l] | prompt_tokens, input_tokens[:l], x) for l in Lg
        """
        Lp = len(prompt_tokens)
        logits = self.predict_logits(None, [prompt_tokens + input_tokens], x)[0]
        logits = logits[Lp - 1:-1].float()                  # (Lg, V): next-token logits over each input pos
        ids = torch.tensor(input_tokens, device=logits.device)
        return logits.log_softmax(dim=-1).gather(1, ids[:, None])[:, 0]         # (Lg,)

    def entropy(self, prompt_tokens, input_tokens, x=None):
        """The next-token distribution's entropy along input_tokens, in one cache-free forward.

        Mirrors log_probs' forward, but reads the whole distribution instead of the taken token: the
        spread of what the model would have said at each position, not how it scored what it did say.

        Args:
            prompt_tokens: list (Lp)
            input_tokens: list (Lg)
            x: (1, Lx, D) | None -- one sample is scored, so N = 1

        Returns:
            (Lg,) H[P(. | prompt_tokens, input_tokens[:l], x)] in nats, for l in Lg
        """
        Lp = len(prompt_tokens)
        logits = self.predict_logits(None, [prompt_tokens + input_tokens], x)[0]
        logits = logits[Lp - 1:-1].float()                  # (Lg, V): next-token logits over each input pos
        log_probs = logits.log_softmax(dim=-1)              # (Lg, V)
        return -(log_probs.exp() * log_probs).sum(dim=-1)   # (Lg,)

    def logits_to_tokens(self, logits, temperature):
        """
        Args:
            logits: (N, V)
            temperature: float, 0.0 = greedy.

        Returns:
            (N, 1) int64
        """
        if temperature == 0.0:
            return logits.argmax(-1, keepdim=True)             # greedy
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)         # (N, 1)

    def tokens_to_texts(self, token_lists):
        """
        Args:
            token_lists: 2D list (N, L), ragged

        Returns:
            list (N) of str, special tokens stripped.
        """
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in token_lists]

    @torch.no_grad()
    def generate(self, token_lists, x=None, max_new_tokens=128, temperature=0.7):
        """
        Args:
            token_lists: 2D list (N, L).
            x: (N, Lx, D) | None, one intervention per sample.
            max_new_tokens: int cap per sample.
            temperature: float, 0.0 = greedy.

        Returns:
            2D list (N, Lg) generated-only ids, ragged; each ends at eos (included) or the cap.
        """
        kv_caches = [DynamicCache() for _ in token_lists]         # fresh per call; the prefill fills them
        logits = self.predict_logits(kv_caches, token_lists, x)   # x rides in here only -- its k/v then lives in the cache
        tokens = self.logits_to_tokens(torch.stack([lg[-1] for lg in logits]), temperature)
        generated_tokens = [[t.item()] for t in tokens]           # seed: first sampled token per sample

        def is_active(i):
            return generated_tokens[i][-1] not in self.eos_token_ids and len(generated_tokens[i]) < max_new_tokens

        active = [i for i in range(len(token_lists)) if is_active(i)]   # a finished sample leaves the batch for good
        while active:
            logits = self.predict_logits([kv_caches[i] for i in active], [[generated_tokens[i][-1]] for i in active])   # x already in the cache
            tokens = self.logits_to_tokens(torch.stack([lg[-1] for lg in logits]), temperature)   # one query/sample
            for j, i in enumerate(active):
                generated_tokens[i].append(tokens[j, 0].item())
            active = [i for i in active if is_active(i)]
        return generated_tokens