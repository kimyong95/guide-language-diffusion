"""Varlen KV-cache pipeline over Qwen3, with an optional per-layer intervention.

A custom attention kernel registered via `AttentionInterface.register` keeps one ragged KV cache per
sample and attends the whole batch in a single `flash_attn_varlen_func` call -- no padding. Qwen3's
own forward stack runs untouched around it, with our varlen state riding in as a `varlen=(...)`
kwarg; `use_cache=False` because we own the cache, and a custom attn-impl name builds no attention
mask, so the kernel owns causality and sample isolation via cu_seqlens + causal=True.

`texts_to_tokens(prompts, n_intervene=Lx)` appends `<|vision_start|><|image_pad|>*Lx<|vision_end|>`
to the user content, and `intervene` overwrites those slots' hidden state with a per-layer x
(H, Lx, D) at every decoder layer's input. They stay ordinary prompt positions with ordinary RoPE,
so x is written once at prefill and its k/v then serves every decode step. `x=None` is a plain
forward, so one Pipeline scores and generates both with and without an intervention.

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
        kv_caches: 1D list (N) of DynamicCache, updated in place.
        key: (1, heads_kv, NL, Dh)
        value: (1, heads_kv, NL, Dh)
        cu_q: (N+1,) int32
        li: int layer index.

    Returns:
        (k_full, v_full), each a 1D list (N) of (Lk_i, heads_kv, Dh).
    """
    k_full, v_full = [], []
    for i, kv in enumerate(kv_caches):                             # per sample (varlen: batch is a python list)
        ki, vi = kv.update(key[:, :, cu_q[i]:cu_q[i + 1]], value[:, :, cu_q[i]:cu_q[i + 1]], li)
        k_full.append(ki[0].transpose(0, 1))                       # (Lk_i, heads_kv, Dh)
        v_full.append(vi[0].transpose(0, 1))
    return k_full, v_full


def varlen_attention(module, query, key, value, attention_mask, **kwargs):
    """Called by Qwen3Attention through the AttentionInterface registry, never by us directly.

    Args:
        module: Qwen3Attention
        query: (1, heads_q, NL, Dh)
        key: (1, heads_kv, NL, Dh)
        value: (1, heads_kv, NL, Dh)
        attention_mask: Always None.
        **kwargs: varlen=(kv_caches: 1D list (N) of DynamicCache, cu_q: (N+1,) int32,
            cu_k: (N+1,) int32), bundled by Pipeline.predict_logits.

    Returns:
        ((NL, heads_q, Dh), None) -- the (attn_output, attn_weights) pair Qwen3Attention expects.
    """
    kv_caches, cu_q, cu_k = kwargs["varlen"]                        # our one bundled kwarg
    k_full, v_full = assemble_kv(kv_caches, key, value, cu_q, module.layer_idx)
    max_q = (cu_q[1:] - cu_q[:-1]).max().item()
    max_k = (cu_k[1:] - cu_k[:-1]).max().item()
    out = flash_attn_varlen_func(
        query[0].transpose(0, 1).contiguous(),                     # (NL, heads_q, Dh)
        torch.cat(k_full), torch.cat(v_full),                      # (sum Lk_i, heads_kv, Dh)
        cu_q, cu_k, max_q, max_k,
        softmax_scale=module.scaling, causal=True)                 # (NL, heads_q, Dh)
    return out, None       # Qwen3Attention.forward reshapes (NL, heads_q, Dh) -> (1, NL, heads_q*Dh)


AttentionInterface.register("varlen_attention", varlen_attention)


# ===================== the loop's view: pack ragged tokens, get logits =======
class Pipeline:
    """ragged tokens + per-sample ragged caches (+ optional intervention) -> logits, generation, scoring (varlen)."""

    INTERVENE_TOKEN = "<|image_pad|>"     # the prompt slot whose hidden state the caller optimizes

    def __init__(self, model_id):
        """
        Args:
            model_id: str HF repo id or local path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        max_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="varlen_attention", device_map="auto", max_memory=max_memory,).eval()
        self.model.requires_grad_(False)                 # only an intervention x ever needs grad, never the weights
        self.device = self.model.device
        self.intervene_token_id = self.tokenizer.convert_tokens_to_ids(self.INTERVENE_TOKEN)

    @contextlib.contextmanager
    def intervene(self, x, positions):
        """Overwrites the hidden state at `positions` with x at every decoder layer's input.

        Layer 0's input is the embedding itself, so one pre-hook per layer covers all H rows of x;
        Qwen3 then computes those positions' k/v (with RoPE) from x as it would for any token. The
        caller locates the slots -- this knows nothing about tokenization, packing or the kernel.

        Args:
            x: (H, Lx, D) | None; row h replaces layer h's input. None registers no hooks.
            positions: (N*Lx,) int64 indices along the sequence axis -- x's Lx slots tile over them,
                one copy per sample, so the count must be a multiple of Lx.

        Yields:
            None; the hooks are live inside the block and removed on exit. Differentiable in x.
        """
        if x is None:
            yield
            return
        N = positions.numel() // x.shape[1]                          # samples sharing this one x

        def pre_hook(layer, args):
            h = args[0]                                              # (N, L, D) residual stream -- (1, NL, D) when packed
            src = einops.repeat(x[layer.self_attn.layer_idx], "lx d -> b (n lx) d", b=h.shape[0], n=N)
            return (h.index_copy(1, positions.to(h.device), src.to(h)),)  # new tensor: no in-place on the graph

        handles = [layer.register_forward_pre_hook(pre_hook) for layer in self.model.model.layers]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def texts_to_tokens(self, prompts, n_intervene=0):
        """
        Args:
            prompts: 1D list (N) of str
            n_intervene: int, INTERVENE_TOKEN slots appended to every prompt's user content.

        Returns:
            2D list (N, L), ragged (no padding)
        """
        marker = f"<|vision_start|>{self.INTERVENE_TOKEN * n_intervene}<|vision_end|>" if n_intervene else ""
        return [self.tokenizer(self.tokenizer.apply_chat_template([{"role": "user", "content": p + marker}], tokenize=False,add_generation_prompt=True, enable_thinking=False)).input_ids for p in prompts]

    def predict_logits(self, kv_caches, input_tokens, x=None):
        """
        Args:
            kv_caches: list (N) of DynamicCache, will be updated inplace.
            input_tokens: 2D list (N, L), ragged; appended per sample.
            x: (H, Lx, D) | None, written to the INTERVENE_TOKEN positions of input_tokens.

        Returns:
            (N, L, V) logits for each input_tokens.
        """
        dev = self.device
        past = [kv.get_seq_length() for kv in kv_caches]    # per-sample length BEFORE append
        q_lens = [len(t) for t in input_tokens]
        input_ids = torch.tensor([list(itertools.chain.from_iterable(input_tokens))], device=dev)
        position_ids = torch.tensor([[p for pa, q in zip(past, q_lens) for p in range(pa, pa + q)]], device=dev)  # (1, NL)
        cu_q = torch.tensor(list(itertools.accumulate(q_lens, initial=0)), dtype=torch.int32, device=dev)  # query offsets
        cu_k = torch.tensor(list(itertools.accumulate((pa + q for pa, q in zip(past, q_lens)), initial=0)), dtype=torch.int32, device=dev)

        # official black-box forward; our varlen state rides in as a kwarg, cache stays ours (use_cache=False)
        bundle = (kv_caches, cu_q, cu_k)
        slots = (input_ids[0] == self.intervene_token_id).nonzero()[:, 0]   # (N*Lx,) into the packed axis
        with self.intervene(x, slots):                      # no-op when x is None
            out = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False, varlen=bundle)
        return list(out.logits[0].split(q_lens))            # [ (L_i, V) ] per sample

    def log_probs(self, prompt_tokens, input_tokens, x=None):
        """Scores input_tokens teacher-forced after prompt_tokens, on a fresh cache.

        Args:
            prompt_tokens: 1D list (Lp)
            input_tokens: 1D list (Lg)
            x: (H, Lx, D) | None

        Returns:
            scalar: log P(input_tokens | prompt_tokens, x)
        """
        Lp = len(prompt_tokens)
        logits = self.predict_logits([DynamicCache()], [prompt_tokens + input_tokens], x)[0]
        logits = logits[Lp - 1:-1].float()                  # (Lg, V): next-token logits over each input pos
        ids = torch.tensor(input_tokens, device=self.device)
        return logits.log_softmax(dim=-1).gather(1, ids[:, None])[:, 0].sum()   # scalar: log P(input | prompt)

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
            1D list (N) of str, special tokens stripped.
        """
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in token_lists]

    @torch.no_grad()
    def generate(self, token_lists, x=None, max_new_tokens=128, temperature=0.7):
        """
        Args:
            token_lists: 2D list (N, L).
            x: (H, Lx, D) | None, intervention.
            max_new_tokens: int cap per sample.
            temperature: float, 0.0 = greedy.

        Returns:
            2D list (N, Lg) generated-only ids, ragged; each ends at eos or the cap.
        """
        N = len(token_lists)
        kv_caches = [DynamicCache() for _ in token_lists]         # fresh per call; the prefill fills them
        logits = self.predict_logits(kv_caches, token_lists, x)   # x rides in here only -- its k/v then lives in the cache
        tokens = self.logits_to_tokens(torch.stack([lg[-1] for lg in logits]), temperature)
        generated_tokens = [[t.item()] for t in tokens]           # seed: first sampled token per sample
        while max(len(g) for g in generated_tokens) < max_new_tokens and not all(g[-1] == self.tokenizer.eos_token_id for g in generated_tokens):
            logits = self.predict_logits(kv_caches, [[g[-1]] for g in generated_tokens])   # x already in the cache
            tokens = self.logits_to_tokens(torch.stack([lg[-1] for lg in logits]), temperature)   # one query/sample
            for i in range(N):                                    # append until (incl.) eos; eos = sample's stop marker
                if generated_tokens[i][-1] != self.tokenizer.eos_token_id:
                    generated_tokens[i].append(tokens[i, 0].item())
        return generated_tokens
