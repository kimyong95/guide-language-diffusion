"""Capture a story's mean hidden state, refine it, and inject it back as a KV cache.

Every stage uses the SAME prompt (PROMPT below).

Stage 1  -- generate a story, read the response's per-layer hidden states x of shape (L, H, D)
            (L = response length, H = num layers, D = dim), and average over L to get
            x_bar of shape (H, D): one hidden vector per layer.
Stage 1b -- starting from x_bar, Adam-optimize a per-layer hidden state x_hat that maximizes the
            log-prob of the Stage-1 response when injected as a KV cache (cf. trainer/optimize-hidden.py).
Stage 2  -- inject x_bar and x_hat (one KV row per layer, mirroring optimize-hidden.build_noisy_cache
            with noise_length L = 1) and generate, comparing against a no-injection baseline.
"""

import einops
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
PROMPT = "讲一个故事。"          # the single prompt shared by every stage
MAX_NEW_TOKENS = 4096
TEMPERATURE = 1.0
TOP_P = 1.0
ENABLE_THINKING = False
SYSTEM_PROMPT = "You are a helpful assistant."
OPTIMIZE_STEPS = 100             # Adam steps used to refine x_bar into x_hat
LEARNING_RATE = 0.1

# --- load model + tokenizer (mirrors trainer/base.py setup_model) ---
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
model.requires_grad_(False)
model.eval()


@torch.no_grad()
def build_prompt_tokens(user_prompt):
    """Tokenize `user_prompt` into a (1, P) input_ids tensor via the chat template (cf. base.py)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=ENABLE_THINKING).to(model.device).input_ids  # (1, P)


@torch.no_grad()
def sample_generate(input_ids, **kwargs):
    """Stochastic decode with the shared sampling settings; `kwargs` passes e.g. past_key_values."""
    return model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        **kwargs,
    )


# The shared prompt, tokenized once. P is the prompt length -- the single injection position.
prompt_tokens = build_prompt_tokens(PROMPT)  # (1, P)
P = prompt_tokens.shape[1]


def build_injected_cache(x):
    """Fresh clean prompt KV cache with ONE injected KV row per layer, built from the per-layer
    hidden state `x` of shape (H, D). Each row is pushed through that layer's real K/V pipeline
    (input_layernorm -> k/v_proj -> k_norm), landing on the model's true K/V manifold. RoPE is
    deliberately NOT applied: the injected row is position-less, so every query attends to it with
    the same (rotation-free) key regardless of where the query sits.
    Differentiable in `x`; wrap the call in torch.no_grad when the graph is not wanted.
    Returns (cache, prompt_logits). Mirrors optimize-hidden.build_noisy_cache with L = 1."""
    with torch.no_grad():
        out = model(prompt_tokens, use_cache=True, logits_to_keep=1)  # only the last row's logits are read
    cache = out.past_key_values
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        dev = attn.k_proj.weight.device
        Dh = attn.head_dim
        xi = x[i][None, None].to(dev, attn.k_proj.weight.dtype)  # (1, 1, D)
        xln = layer.input_layernorm(xi)  # RMSNorm is scale-invariant, so only the direction of x matters
        k = attn.k_norm(einops.rearrange(attn.k_proj(xln), "b l (h d) -> b l h d", d=Dh))  # k_norm over head_dim
        k = einops.rearrange(k, "b l h d -> b h l d")               # (1, Hkv, 1, Dh)
        v = einops.rearrange(attn.v_proj(xln), "b l (h d) -> b h l d", d=Dh)  # (1, Hkv, 1, Dh)
        cl = cache.layers[i]
        cl.keys = torch.cat([cl.keys, k.to(cl.keys.device)], dim=2)
        cl.values = torch.cat([cl.values, v.to(cl.values.device)], dim=2)
    return cache, out.logits


@torch.no_grad()
def generate_with_injection(x):
    """Generate from the shared prompt with one injected KV row per layer built from `x`. The first
    token is seeded from the prompt-only logits (it cannot attend to the injected row, which sits at
    cache index P); the single placeholder stands in for that row and is never embedded.
    The injected row carries no RoPE, so it consumes no position: the response is placed at
    positions P, P+1, ... -- right after the prompt rather than after P+1 cache slots."""
    cache, prompt_logits = build_injected_cache(x)
    first_token = prompt_logits[:, -1:, :].argmax(-1)  # (1, 1)
    placeholder = prompt_tokens.new_full((1, 1), tokenizer.pad_token_id)
    input_ids = torch.cat([prompt_tokens, placeholder, first_token], dim=1)  # (1, P + 1 + 1)
    # prompt -> 0..P-1, placeholder -> P (never read: it is sliced off with the cached prefix),
    # y_0 -> P. `generate` extends this by +1 per step, so y_j lands at P + j.
    position_ids = torch.arange(P + 2, device=model.device).clamp(max=P)[None]  # (1, P + 2)
    out = sample_generate(input_ids, past_key_values=cache, position_ids=position_ids)
    return out[0, P + 1:]  # response tokens (drop prompt + placeholder)


def optimize_hidden(x_init, target_tokens):
    """Adam-optimize a per-layer hidden state, starting from `x_init` (H, D), to maximize the
    log-prob of `target_tokens` when injected as a KV cache. Mirrors optimize-hidden.training_step,
    minus the policy-gradient advantage weight: the objective is simply the log-likelihood of one
    fixed target sequence (the Stage-1 response). Returns the optimized x_hat (H, D), detached."""
    x = x_init.clone().detach().requires_grad_(True)  # (H, D) fp32 leaf to optimize
    optimizer = torch.optim.Adam([x], lr=LEARNING_RATE)
    targets = target_tokens.to(model.device)  # (T,) y_0 .. y_{T-1}
    for step in range(1, OPTIMIZE_STEPS + 1):
        optimizer.zero_grad()
        cache, _ = build_injected_cache(x)  # fresh each step; the forward appends the response K/V to it
        # y_j sits at position P + j: the position-less injected row occupies a cache slot but no position.
        positions = torch.arange(P, P + targets.shape[0], device=model.device)[None]  # (1, T)
        logits = model(targets[None], position_ids=positions, past_key_values=cache, use_cache=True).logits  # (1, T, vocab)
        # logits[:, j] predicts y_{j+1}; y_0 is dropped -- it is seeded from the prompt-only logits
        # and cannot attend to the injected row, so its gradient w.r.t. x is exactly zero.
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        token_log_probs = log_probs.gather(-1, targets[None, 1:, None]).squeeze(-1).squeeze(0)  # (T-1,)
        loss = -token_log_probs.mean()
        loss.backward()
        optimizer.step()
        print(f"  optimize step {step:2d}/{OPTIMIZE_STEPS}   loss = {loss.item():.4f}")
    return x.detach()


# ======================= Stage 1: generate a story, capture x_bar =======================
full_ids = sample_generate(prompt_tokens)  # (1, P + T)
response_tokens = full_ids[0, P:]           # generated story y_0 .. y_{T-1}
print("=" * 80)
print("STAGE 1 -- generated story:")
print(tokenizer.decode(response_tokens, skip_special_tokens=True))

# Teacher-forced forward over the full sequence so the response tokens attend to the prompt; keep the
# response region only. hidden_states is a tuple of H+1 tensors: indices 0..H-1 are the inputs to each
# layer (what input_layernorm consumes), so we drop the final normed output (index H).
with torch.no_grad():
    out = model(full_ids, output_hidden_states=True)
H = model.config.num_hidden_layers
# hidden_states[i] may live on a different GPU under device_map="auto"; move to one device to stack.
hs = torch.stack([out.hidden_states[i][0, P:, :].to(model.device) for i in range(H)], dim=1)  # (T, H, D)
x_bar = hs.float().mean(dim=0)  # (H, D) -- mean over the response length L

# ======================= Stage 1b: refine x_bar into x_hat =======================
print("=" * 80)
print(f"STAGE 1b -- optimizing x_hat from x_bar for {OPTIMIZE_STEPS} Adam steps:")
x_hat = optimize_hidden(x_bar, response_tokens)

# ======================= Stage 2: generate with each hidden state injected =======================
baseline_tokens = sample_generate(prompt_tokens)[0, P:]
print("=" * 80)
print("STAGE 2 baseline -- no injection:")
print(tokenizer.decode(baseline_tokens, skip_special_tokens=True))

print("=" * 80)
print("STAGE 2 -- x_bar injected:")
print(tokenizer.decode(generate_with_injection(x_bar), skip_special_tokens=True))

print("=" * 80)
print("STAGE 2 -- x_hat injected:")
print(tokenizer.decode(generate_with_injection(x_hat), skip_special_tokens=True))
