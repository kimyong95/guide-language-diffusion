# accelerate launch --num_processes 1 test-sampling.py

import shutil

import torch
from accelerate import Accelerator

from pipeline import DiffusionGemmaPipeline

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
PROMPT = "Why is the sky blue?"
NUM_INFERENCE_STEPS = 48
MAX_TOKENS = 4096   # sliding window: stop after this many committed tokens

accelerator = Accelerator()

# Shard this process's full model across its GPU stride
max_memory = {i: torch.cuda.get_device_properties(i).total_memory
              for i in range(accelerator.process_index, torch.cuda.device_count(), accelerator.num_processes)}

pipeline = DiffusionGemmaPipeline(
    MODEL_ID,
    canvas_length=256,
    entropy_bound=0.1,
    t_min=0.4,
    t_max=0.8,
    device=accelerator.device,
    device_map="auto",
    max_memory=max_memory,
)
pipeline.model.eval()

L = pipeline.canvas_length
V = pipeline.vocab_size
device = pipeline.device

# set_timesteps also sets scheduler.num_inference_steps, which temperature() needs.
pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
timesteps = torch.full((L,), NUM_INFERENCE_STEPS, device=device, dtype=torch.long)  # per-position lives

prompt_tokens = pipeline.build_prompt_tokens(PROMPT, enable_thinking=True)
kv_cache = pipeline.build_kv_cache(prompt_tokens)

# --- Experimental sliding-window sampling -------------------------------------
xt_logits = None                                 # self-conditioning off on the first step
xt_tokens = pipeline.sample_init_tokens()[None]  # (1, L) fully-noised canvas ~ Uniform(V)

committed = []       # list of (k,) committed token-id tensors
n_committed = 0
while n_committed < MAX_TOKENS:
    xt_logits, finished = pipeline.model_predict(xt_tokens, xt_logits, timesteps, kv_cache)  # (L, V), (L,)
    timesteps = torch.clamp(timesteps - 1, min=0)         # age every position by one step, floor at 0

    k = int(finished.long().cumprod(dim=0).sum())   # length of the leading all-finished prefix

    if k > 0:
        commit = pipeline.argmax_logits_to_tokens(xt_logits[:k])  # (k,) clean tokens
        committed.append(commit)
        n_committed += k
        if torch.isin(commit, pipeline.eos_token_id).any():
            break                                    # EOS reached: stop before caching this commit
        kv_cache = pipeline.build_kv_cache(commit[None], kv_cache)  # grow cache by k -> decoder positions auto-slide

        # slide the window: drop the k committed positions, append k fresh hot positions
        xt_logits = torch.cat([xt_logits[k:], torch.ones(k, V, device=device, dtype=xt_logits.dtype)], dim=0)
        timesteps = torch.cat([timesteps[k:], torch.full((k,), NUM_INFERENCE_STEPS, device=device, dtype=torch.long)], dim=0)

    # renoise for the next step; fresh tail positions (uniform logits) renoise to Uniform(V)
    xt_tokens = pipeline.sample_logits_to_tokens(xt_logits)[None]  # (1, L)

    if accelerator.is_main_process and committed:
        cols = shutil.get_terminal_size().columns
        live = pipeline.processor.decode(torch.cat(committed), skip_special_tokens=False).replace("\n", " ")
        print("\r" + live[-(cols - 1):] + "\033[K", end="", flush=True)

if accelerator.is_main_process:
    print()  # finish the in-place line
    text = pipeline.processor.decode(torch.cat(committed), skip_special_tokens=False)
    print(text)

accelerator.wait_for_everyone()
accelerator.end_training()
