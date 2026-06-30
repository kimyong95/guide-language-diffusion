from functools import partial

import torch

from pipeline import DiffusionGemmaPipeline, token_entropy

B = 2 # number of samples to generate
PROMPT = "tell me something"
NUM_INFERENCE_STEPS = 48
TARGET_LAYER_ID = 14  # decoder layer whose input we record / replace
MUTATE_RATIO = 0.5  # fraction of (highest-entropy) init-token positions to flip to the 2nd-best token

# Build the pipeline (defaults mirror config/flow-guide.py).
pipeline = DiffusionGemmaPipeline(
    "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic",
    gen_length=256,
    entropy_bound=0.1,
    t_min=0.4,
    t_max=0.8,
)
pipeline.model.eval()

# Timestep schedule (N..1) and prompt KV cache are set up once, shared across samples.
timesteps = pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=pipeline.device)
pipeline.set_prompt(PROMPT)


# --- hidden-state injection (mirrors Trainer.extended_decoder_layer_forward in flow-guide.py) ---
# When enabled, replace TARGET_LAYER_ID's block input with `inject["hs"]` ((L, D) for the
# current sample). Everything from that layer onward then only depends on the injected tensor.
inject = {"enabled": False, "hs": None}


@torch.no_grad()
def extended_decoder_layer_forward(self, hidden_states, *args, original_forward=None, **kwargs):
    x = hidden_states  # (n, L, D) block input
    if inject["enabled"] and self.layer_idx == TARGET_LAYER_ID:
        x = inject["hs"][None].to(device=x.device, dtype=x.dtype)  # (1, L, D), replace
    return original_forward(x, *args, **kwargs)


for block in pipeline.model.model.decoder.layers:
    block.forward = partial(extended_decoder_layer_forward, block, original_forward=block.forward)


# Pass 1: plain generation; record each sample's layer-14 input at the last timestep.
hidden_states_list = []  # (L, D) per sample
tokens_list = []  # (L,) per sample
for b in range(B):
    xt_logits = None  # (L, V); None on step 0
    for timestep in timesteps:
        xt_logits, hidden_states, early_stop = pipeline.model_predict_step(xt_logits, timestep)
        if early_stop:
            print(f"Early stopping at timestep {timestep} for sample {b}.")
            break
    tokens_list.append(pipeline.argmax_logits_to_tokens(xt_logits))  # (L,)
    hidden_states_list.append(hidden_states[TARGET_LAYER_ID].clone())  # input to layer TARGET_LAYER_ID

    text = pipeline.argmax_logits_to_text(xt_logits)
    print(f"===== sample {b} (original), spent {timesteps[0] - timestep} steps =====")
    print(text)

# Pass 2: mixing
# Generate one sample, injecting the average of the two recorded layer-TARGET_LAYER_ID hidden
# states, but only at the first generation step. Every later step runs unmodified.
inject["hs"] = torch.stack(hidden_states_list).mean(dim=0)  # (L, D), average of the two
xt_logits = None  # (L, V); None on step 0
for step, timestep in enumerate(timesteps):
    inject["enabled"] = step == 0  # inject only at the first generation step
    xt_logits, hidden_states, early_stop = pipeline.model_predict_step(xt_logits, timestep)
    if early_stop:
        print(f"Early stopping at timestep {timestep}.")
        break
inject["enabled"] = False
text = pipeline.argmax_logits_to_text(xt_logits)
print(f"===== averaged-injection, spent {timesteps[0] - timestep} steps =====")
print(text)

# Pass 3: mutate
# Identical procedure to the pipeline (model_predict_step), written raw so step 0 can be
# seeded with the pass-1 argmax tokens as the prior (instead of a uniform-random canvas).
# Everything else matches: self-conditioning = the previous step's logits (None on step 0),
# predict logits, then entropy-renoise the canvas (copied from sample_logits_to_tokens).
for b in range(B):
    xt_tokens = tokens_list[b]  # (L,) prior: pass-1 generated tokens
    xt_logits = None            # previous step's logits, for the early-stop check
    
    # Mutate the init tokens: one forward over the prior (self-conditioning None) -> logits;
    # keep argmax for the most-confident (1 - MUTATE_RATIO) positions and take the 2nd-best
    # token for the highest-entropy MUTATE_RATIO positions, then continue the loop with them.
    with torch.no_grad():
        out = pipeline.model(
            input_ids=None,
            past_key_values=pipeline.kv_cache,
            decoder_position_ids=pipeline.dec_pos,
            decoder_input_ids=xt_tokens[None],  # (1, L)
            self_conditioning_logits=None,
            output_hidden_states=False,
        )
    logits = out.logits[0]  # (L, V)
    top2 = torch.topk(logits, k=2, dim=-1).indices  # (L, 2): [argmax, 2nd argmax]
    n_mutate = int(pipeline.gen_length * MUTATE_RATIO)
    mutate_pos = torch.argsort(token_entropy(logits), descending=True)[:n_mutate]  # highest-entropy positions
    xt_tokens = top2[:, 0].clone()                # argmax for the confident positions
    xt_tokens[mutate_pos] = top2[mutate_pos, 1]   # 2nd argmax for the mutated positions

    print(f"===== sample {b} (mutated init tokens) =====")
    print(pipeline.processor.decode(xt_tokens, skip_special_tokens=True))
    
    for timestep in timesteps:
        with torch.no_grad():
            out = pipeline.model(
                input_ids=None,
                past_key_values=pipeline.kv_cache,
                decoder_position_ids=pipeline.dec_pos,
                decoder_input_ids=xt_tokens[None],  # (1, L)
                self_conditioning_logits=xt_logits,  # previous step's logits; None on step 0
                output_hidden_states=False,
            )
        xt_logits_next = pipeline.scheduler.temperature(out.logits, timstep=timestep)[0]  # (L, V)

        if xt_logits is not None and pipeline.early_stop(xt_logits, xt_logits_next):
            xt_logits = xt_logits_next
            print(f"Early stopping at timestep {timestep} for sample {b}.")
            break
        xt_logits = xt_logits_next

        # entropy-based renoise (copied from pipeline.sample_logits_to_tokens): accept the
        # lowest-entropy positions within entropy_bound, renoise the rest to Uniform(V).
        entropy = token_entropy(xt_logits)  # (L,)
        sorted_entropy, order = torch.sort(entropy, descending=False)
        preceding_sum = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
        accept = torch.empty_like(entropy, dtype=torch.bool)
        accept[order] = preceding_sum <= pipeline.entropy_bound  # (L,)
        probs = torch.softmax(xt_logits, dim=-1, dtype=torch.float32)  # (L, V)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Categorical(P_t)
        noise = torch.randint(pipeline.vocab_size, entropy.shape, device=xt_logits.device)  # Uniform(V)
        xt_tokens = torch.where(accept, sampled, noise)  # (L,) canvas for the next step

    text = pipeline.argmax_logits_to_text(xt_logits)
    print(f"===== sample {b} (mutated), spent {timesteps[0] - timestep} steps =====")
    print(text)
