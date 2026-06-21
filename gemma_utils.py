from accelerate import init_empty_weights
import torch
from transformers import DiffusionGemmaForBlockDiffusion, AutoConfig

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

def module_bytes(module, param_dtype=torch.bfloat16):
    """Bytes a module needs: params counted at `param_dtype`, buffers at their own dtype."""
    bytes_per_param = torch.finfo(param_dtype).bits // 8
    total = sum(p.numel() * bytes_per_param for p in module.parameters(recurse=True))
    total += sum(b.numel() * b.element_size() for b in module.buffers(recurse=True))
    return total


def build_device_map(headroom=0.8):
    """Distribute DiffusionGemma across GPUs, spilling to CPU only when GPUs are full.

    Non-layer modules (embeddings, norms, vision, lm_head, ...) live on the first GPU. The
    tied encoder/decoder layer pairs are packed GPU-by-GPU until a GPU would exceed
    `headroom` of its memory, then the next GPU, then CPU once all GPUs are full.

    The map is a disjoint partition with NO "" catch-all: accelerate attaches a
    place_submodules hook to any GPU key that is an ancestor of an offloaded layer, which
    would eagerly pull the whole model onto one GPU (OOM). See accelerate hooks.py:656 and
    303-305.
    """
    config = AutoConfig.from_pretrained(MODEL_ID)
    with init_empty_weights():
        skeleton = DiffusionGemmaForBlockDiffusion(config)  # meta device, no real memory
    
    enc_layers_str = "model.encoder.language_model.layers"
    dec_layers_str = "model.decoder.layers"

    enc_layers = skeleton.model.encoder.language_model.layers
    dec_layers = skeleton.model.decoder.layers

    skeleton.tie_weights()
    base_bytes = module_bytes(skeleton) - module_bytes(enc_layers)

    budget = lambda g: int(headroom * torch.cuda.get_device_properties(g).total_memory)

    # Non-layer modules -> first GPU, at the layer stacks' sibling granularity so no GPU
    # key is an ancestor of an offloaded layer.
    device_map = {}
    for name, module in skeleton.named_modules():
        if enc_layers_str.startswith(name) or name.startswith(enc_layers_str) or dec_layers_str.startswith(name) or name.startswith(dec_layers_str):
            continue  # skip layers, handled below
        device_map[name] = 0  # first GPU

    # Layer pairs -> packed GPU-by-GPU, spilling to CPU when GPUs are full.
    device, used = 0, base_bytes
    for i in range(len(enc_layers)):
        layer_bytes = module_bytes(enc_layers[i])
        if device != "cpu" and used + layer_bytes > budget(device):
            device = device + 1 if device + 1 < torch.cuda.device_count() else "cpu"  # next GPU, else spill to CPU
            used = 0
        used += layer_bytes
        device_map[f"{enc_layers_str}.{i}"] = device
        device_map[f"{dec_layers_str}.{i}"] = device

    del skeleton
    return device_map