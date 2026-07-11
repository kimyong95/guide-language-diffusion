# accelerate launch --num_processes 1 test-pipeline.py
#
# Same feature as test-gemma.py, but drives DiffusionGemmaPipeline (block diffusion) instead of
# model.generate. Each process shards its full model across its GPU stride via max_memory; run
# with --num_processes 1 to shard one model across the visible GPUs, or more for DP replicas.

import time

import torch
from accelerate import Accelerator

from pipeline import DiffusionGemmaPipeline

acc = Accelerator()

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
PROMPT = "Why is the sky blue?"
NUM_INFERENCE_STEPS = 48
MAX_BLOCKS = 24

max_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(acc.process_index, torch.cuda.device_count(), acc.num_processes)}

pipeline = DiffusionGemmaPipeline(
    MODEL_ID,
    canvas_length=256,
    entropy_bound=0.1,
    t_min=0.4,
    t_max=0.8,
    device=acc.device,
    device_map="auto",
    max_memory=max_memory,
)
pipeline.model.eval()

timesteps = pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=acc.device)
prompt_tokens = pipeline.build_prompt_tokens(PROMPT, enable_thinking=True)
kv_cache = pipeline.build_kv_cache(prompt_tokens)
generated = []
for block in range(MAX_BLOCKS):
    xt_logits = None
    xt_tokens = pipeline.sample_init_tokens()[None]
    for timestep in timesteps:
        xt_logits, finished = pipeline.model_predict(xt_tokens, xt_logits, timestep, kv_cache)
        xt_tokens = pipeline.sample_logits_to_tokens(xt_logits)[None]
        if finished[-1]:
            break

    canvas = pipeline.argmax_logits_to_tokens(xt_logits)
    generated.append(canvas)
    kv_cache = pipeline.build_kv_cache(canvas[None], kv_cache)
    if torch.isin(canvas, pipeline.eos_token_ids).any():
        break

gen_tokens = pipeline.strip_thinking_tokens(torch.cat(generated))
text = pipeline.processor.decode(gen_tokens, skip_special_tokens=True)

if acc.is_main_process:
    print(text)