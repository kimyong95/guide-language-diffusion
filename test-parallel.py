# accelerate launch --num_processes 4 test-parallel.py 

import torch
import torch.distributed as dist
from accelerate import Accelerator

from pipeline import DiffusionGemmaPipeline, early_stop

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
PROMPT = "Why is the sky blue?"
NUM_INFERENCE_STEPS = 48
MAX_BLOCKS = 24
TP_SIZE = 2

tp_plan = {
    "model.encoder.language_model.layers.*.experts.gate_up_proj": "packed_colwise",
    "model.encoder.language_model.layers.*.experts.down_proj": "rowwise",
    "model.encoder.language_model.layers.*.experts": "moe_tp_experts",
    "model.decoder.layers.*.experts.gate_up_proj": "packed_colwise",
    "model.decoder.layers.*.experts.down_proj": "rowwise",
    "model.decoder.layers.*.experts": "moe_tp_experts",
}

accelerator = Accelerator()

DP_SIZE = accelerator.num_processes // TP_SIZE
device_mesh = dist.init_device_mesh("cuda", (DP_SIZE, TP_SIZE), mesh_dim_names=("dp", "tp"))

tp_group = device_mesh["tp"].get_group()
tp_root = dist.get_process_group_ranks(tp_group)[0]  # this rank's TP-group root, as a *global* rank

pipeline = DiffusionGemmaPipeline(
    MODEL_ID,
    gen_length=256,
    entropy_bound=0.1,
    t_min=0.4,
    t_max=0.8,
    tp_plan=tp_plan,
    device_mesh=device_mesh,
)
pipeline.model.eval()

L = pipeline.gen_length
device = pipeline.device


timesteps = pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)

prompt_tokens = pipeline.build_prompt_tokens(PROMPT, enable_thinking=True)
kv_cache = pipeline.build_kv_cache(prompt_tokens)

generated = []
for block in range(MAX_BLOCKS):
    xt_logits = None
    xt_tokens = pipeline.sample_init_tokens()[None]
    dist.broadcast(xt_tokens, src=tp_root, group=tp_group)
    for timestep in timesteps:
        xt_logits, _, finished = pipeline.model_predict(xt_tokens, xt_logits, timestep, kv_cache)
        xt_tokens = pipeline.sample_logits_to_tokens(xt_logits)[None]
        dist.broadcast(xt_tokens, src=tp_root, group=tp_group)

        if finished[-1]:
            break

    canvas = pipeline.argmax_logits_to_tokens(xt_logits)
    generated.append(canvas)
    if torch.isin(canvas, pipeline.eos_token_id).any():
        break
    kv_cache = pipeline.build_kv_cache(canvas[None], kv_cache)

gen_tokens = pipeline.strip_thinking_tokens(torch.cat(generated))
text = pipeline.processor.decode(gen_tokens, skip_special_tokens=True)
print(f"[{accelerator.process_index}]: {text}")

accelerator.wait_for_everyone()
accelerator.end_training()