"""Varlen KV-cache pipeline over Qwen3, with optional position-less guidance rows.

We register custom *varlen* attention kernels via `AttentionInterface.register`, then run Qwen3's
OWN forward stack (Qwen3ForCausalLM/Model/DecoderLayer/Attention) as black boxes. The registered fn
is the only custom compute: it appends each row's new k/v into a ragged per-row KV cache and attends
the whole batch in one `flash_attn_varlen_func` call -- no padding.

How the plumbing works (all verified against transformers source):
  - custom kwargs (`varlen=(...)`) thread through every forward via **kwargs down to our fn;
  - `use_cache=False` keeps the model's own (rectangular) cache update from running -- we own the
    cache and pass it in the `varlen` kwarg instead;
  - a *custom* attn-impl name isn't in the mask registry, so the model builds NO attention mask
    (our fn owns causality + row isolation via cu_seqlens + causal=True).

Guidance (mirrors the old extend_transformers, no dependency): `varlen_attention_guidance`
FRONT-prepends a few *position-less* KV rows -- built from a per-layer hidden state `x` by
`hidden_states_to_kv_caches` (the model's own K/V pipeline minus RoPE) -- so every query attends
them. `guidance=None` falls back to the plain path, so one guided Pipeline can score/generate both
unguided and guided (needed by the optimize loop). The impl is fixed at load; pick the guided kernel
with `Pipeline(model_id, "varlen_attention_guidance")`.

Full attention only (a sliding window would slide the front rows out of view); relies on flash's
bottom-right causal alignment, so guidance rows must be front-placed.
"""

import itertools

import einops
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, AttentionInterface
from flash_attn import flash_attn_varlen_func


# ===================== our attention kernels (the only custom compute) =======
def assemble_kv(kv_caches, key, value, cu_q, li):
    """Append each row's new k/v into its own cache; return per-row full k/v as (len_i, Hkv, D) lists.
    The shared 'grow the ragged cache' step behind both kernels below."""
    k_full, v_full = [], []
    for i, kv in enumerate(kv_caches):                             # per row (varlen: batch is a python list)
        ki, vi = kv.update(key[:, :, cu_q[i]:cu_q[i + 1]], value[:, :, cu_q[i]:cu_q[i + 1]], li)
        k_full.append(ki[0].transpose(0, 1))                       # (len_i, Hkv, D)
        v_full.append(vi[0].transpose(0, 1))
    return k_full, v_full


def varlen_attn_kernel(module, query, k_full, v_full, cu_q, cu_k):
    """One varlen kernel over the packed batch. GQA native (no repeat_kv); causal aligns for q_len<k_len.
    max_q/max_k are derived from cu_q/cu_k (a small host sync) so the bundle need only carry cu_q/cu_k."""
    max_q = (cu_q[1:] - cu_q[:-1]).max().item()
    max_k = (cu_k[1:] - cu_k[:-1]).max().item()
    out = flash_attn_varlen_func(
        query[0].transpose(0, 1).contiguous(),                     # (total_q, Hq, D)
        torch.cat(k_full), torch.cat(v_full),                      # (total_k, Hkv, D)
        cu_q, cu_k, max_q, max_k,
        softmax_scale=module.scaling, causal=True)                 # (total_q, Hq, D)
    return out, None       # Qwen3Attention.forward reshapes (total_q, Hq, D) -> (1, total_q, Hq*D)


def varlen_attention(module, query, key, value, attention_mask, **kwargs):
    """HF attention interface: q/k/v are (1, H, total_q, D), already projected + q/k-norm + RoPE
    by Qwen3Attention. Append to per-row ragged caches, then one varlen kernel over the batch."""
    kv_caches, cu_q, cu_k = kwargs["varlen"]                        # our one bundled kwarg
    k_full, v_full = assemble_kv(kv_caches, key, value, cu_q, module.layer_idx)
    return varlen_attn_kernel(module, query, k_full, v_full, cu_q, cu_k)


def varlen_attention_guidance(module, query, key, value, attention_mask, **kwargs):
    """Like varlen_attention, but FRONT-prepends this layer's position-less guidance rows to every
    row's k/v (so every query attends them) and widens each row's key span by L. Front + causal lets
    query j see all L rows + its causal prefix. `guidance is None` -> plain path (unguided). The rows
    are never stored in kv_caches -- they ride in fresh each forward, built by hidden_states_to_kv_caches."""
    kv_caches, cu_q, cu_k, guidance = kwargs["varlen"]             # guidance added to the bundle (may be None)
    k_full, v_full = assemble_kv(kv_caches, key, value, cu_q, module.layer_idx)
    if guidance is not None:
        g = guidance.layers[module.layer_idx]                      # this layer's rows, (1, Hkv, L, Dh)
        gk, gv = g.keys[0].transpose(0, 1), g.values[0].transpose(0, 1)       # (L, Hkv, D)
        k_full = [torch.cat([gk, k]) for k in k_full]              # front placement: guidance ahead of prefix
        v_full = [torch.cat([gv, v]) for v in v_full]
        L = gk.shape[0]
        cu_k = cu_k + torch.arange(len(kv_caches) + 1, device=cu_k.device, dtype=torch.int32) * L   # +L keys/row
    return varlen_attn_kernel(module, query, k_full, v_full, cu_q, cu_k)   # max_k re-derived from cu_k


AttentionInterface.register("varlen_attention", varlen_attention)
AttentionInterface.register("varlen_attention_guidance", varlen_attention_guidance)


# ===================== the loop's view: pack ragged tokens, get logits =======
class Pipeline:
    """ragged tokens + per-row ragged caches (+ optional guidance) -> logits, generation, scoring (varlen)."""

    def __init__(self, model_id, attn_implementation="varlen_attention"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        max_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation=attn_implementation, device_map="auto", max_memory=max_memory,).eval()
        self.model.requires_grad_(False)                 # only a guidance x ever needs grad, never the weights
        self.device = self.model.device
        self.guided = "guidance" in attn_implementation  # whether the loaded kernel expects a guidance slot

    def hidden_states_to_kv_caches(self, x):
        """Per-layer hidden state x (H, L, D) -> a DynamicCache of L position-less guidance rows.
        Runs each layer's OWN K/V pipeline -- input_layernorm -> k_proj -> k_norm, v_proj -- exactly
        as Qwen3Attention does (modeling_qwen3.py:264-265), but SKIPS RoPE (:268): the rows carry no
        position, so every query attends them identically. Each layer's rows sit on that layer's
        device (device_map='auto' safe). Differentiable in x (the optimize loop backprops through here)."""
        kv_cache = DynamicCache()
        for layer in self.model.model.layers:
            attn = layer.self_attn
            xln = layer.input_layernorm(x[attn.layer_idx][None].to(attn.k_proj.weight.device, attn.k_proj.weight.dtype))
            k = attn.k_norm(einops.rearrange(attn.k_proj(xln), "b l (h d) -> b l h d", d=attn.head_dim))  # norm over head_dim
            k = einops.rearrange(k, "b l h d -> b h l d")
            v = einops.rearrange(attn.v_proj(xln), "b l (h d) -> b h l d", d=attn.head_dim)
            kv_cache.update(k, v, attn.layer_idx)                     # (1, Hkv, L, Dh) per layer; no RoPE
        return kv_cache

    def texts_to_tokens(self, prompts):
        """prompts -> ragged list[list[int]] (each its own length, no padding)."""
        return [self.tok(self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,add_generation_prompt=True, enable_thinking=False)).input_ids for p in prompts]

    def logits_forward(self, kv_caches, new_tokens, kv_caches_guidance=None):
        """Packed varlen forward -> per-row logits: a list of (q_len_i, vocab), one per row; advances
        kv_caches. Grad flows unless the caller is under torch.no_grad -- generation is no_grad, the
        optimize loop is not."""
        dev = self.device
        past = [kv.get_seq_length() for kv in kv_caches]    # per-row length BEFORE append
        q_lens = [len(t) for t in new_tokens]
        input_ids = torch.tensor([list(itertools.chain.from_iterable(new_tokens))], device=dev)
        position_ids = torch.tensor([[p for pa, q in zip(past, q_lens) for p in range(pa, pa + q)]], device=dev)  # (1, tot_q)
        cu_q = torch.tensor(list(itertools.accumulate(q_lens, initial=0)), dtype=torch.int32, device=dev)  # query offsets
        cu_k = torch.tensor(list(itertools.accumulate((pa + q for pa, q in zip(past, q_lens)), initial=0)), dtype=torch.int32, device=dev)

        # official black-box forward; our varlen state rides in as a kwarg, cache stays ours (use_cache=False)
        bundle = (kv_caches, cu_q, cu_k)
        if self.guided:
            bundle += (kv_caches_guidance,)             # guided kernel always gets the slot (may be None)
        out = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False, varlen=bundle)
        return list(out.logits[0].split(q_lens))            # [ (q_len_i, vocab) ] per row

    @torch.no_grad()
    def predict_logits(self, kv_caches, input_tokens, kv_caches_guidance=None):
        """last-token logits per row -> (batch, vocab), for sampling. Packs all rows into one varlen
        sequence (no padding); advances caches."""
        logits = self.logits_forward(kv_caches, input_tokens, kv_caches_guidance)
        return torch.stack([lg[-1] for lg in logits])       # each row's last token -> (batch, vocab)

    def log_probs(self, prompt_tokens, input_tokens, kv_caches_guidance=None):
        """log P_theta(input_tokens | prompt_tokens, guidance): the summed next-token log-probability
        of input_tokens teacher-forced after prompt_tokens (one packed row, fresh cache). A scalar,
        differentiable -- backprop reaches a guidance x through hidden_states_to_kv_caches, so this is
        a training signal, not for sampling. Sum (not mean) = the sequence log-prob; `.sum()` over the
        per-token log-probs kept just before it if you want them per-token instead."""
        P = len(prompt_tokens)
        logits = self.logits_forward([DynamicCache()], [prompt_tokens + input_tokens], kv_caches_guidance)[0]
        logits = logits[P - 1:-1].float()                   # (T, vocab): next-token logits over each input pos
        ids = torch.tensor(input_tokens, device=self.device)
        return logits.log_softmax(dim=-1).gather(1, ids[:, None])[:, 0].sum()   # scalar: log P(input | prompt)

    def tokens_to_caches(self, token_lists, kv_caches_guidance=None):
        """ragged prompts -> (next-token logits, one fresh KV cache per row the caller owns)."""
        kv_caches = [DynamicCache() for _ in token_lists]
        logits = self.predict_logits(kv_caches, token_lists, kv_caches_guidance)
        return logits, kv_caches                     # (batch, vocab), list[DynamicCache]

    def logits_to_tokens(self, logits, temperature):
        """next-token logits (batch, vocab) -> sampled tokens (batch, 1)."""
        if temperature == 0.0:
            return logits.argmax(-1, keepdim=True)             # greedy
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)         # (batch, 1)

    def tokens_to_texts(self, token_lists):
        """ragged list[list[int]] -> list[str] (decoded, special tokens stripped)."""
        return [self.tok.decode(t, skip_special_tokens=True) for t in token_lists]

    def generate(self, token_lists, kv_caches_guidance=None, max_new_tokens=128, temperature=0.7):
        """ragged prompts -> ragged generated tokens per row. Owns the sampling loop: prefill, then
        varlen decode a token per row until each hits eos or max_new_tokens. If kv_caches_guidance is
        given, those position-less rows are injected on every forward (prefill + each step)."""
        batch = len(token_lists)
        logits, kv_caches = self.tokens_to_caches(token_lists, kv_caches_guidance)
        tokens = self.logits_to_tokens(logits, temperature)
        generated_tokens = [[t.item()] for t in tokens]           # seed: first sampled token per row
        while max(len(g) for g in generated_tokens) < max_new_tokens and not all(g[-1] == self.tok.eos_token_id for g in generated_tokens):
            logits = self.predict_logits(kv_caches, [[g[-1]] for g in generated_tokens], kv_caches_guidance)
            tokens = self.logits_to_tokens(logits, temperature)
            for i in range(batch):                                # append until (incl.) eos; eos = row's stop marker
                if generated_tokens[i][-1] != self.tok.eos_token_id:
                    generated_tokens[i].append(tokens[i, 0].item())
        return generated_tokens
