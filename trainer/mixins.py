import random

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset (copied & adapted from finetune-stable-diffusion/trainer/mixins.py)
# ---------------------------------------------------------------------------

class DistributedSubsampleDataset(Dataset):
    """Wraps an in-memory list of items and exposes a deterministic per-epoch subsample.

    Each epoch, subsample() draws m unique items and repeats each k times, giving m*k entries. The
    repeated block of an item is one GRPO group. Distributed training: every process reseeds from the
    same base_seed+epoch so the index list is identical across ranks and per-device batches stay
    consistent after gather().

    Unlike the finetune-stable-diffusion original (which loads a txt file of prompts), this takes the
    item list directly so the task can own the data (e.g. GSM8K (Q, A) dicts).
    """

    def __init__(self, all_data, B, G, m, b_max, base_seed=0):
        # N      : total number of items
        # B      : total samples per epoch (across all GPUs)
        # m      : number of unique items sampled per epoch    (m == -1 → m = B; capped at N)
        # k      : repetitions per item per epoch              (B = m*k)
        # G      : number of GPUs (processes)
        # B_i    : total samples per epoch per GPU             (B_i = B/G)
        # b_max  : max batch size per GPU
        # b      : actual batch size per GPU                   (b = min(b_max, B/G))
        # K      : number of batches per epoch per GPU         (K = B_i//b)

        self.all_data = all_data
        self.N = len(self.all_data)
        self.base_seed = base_seed
        self.B = B if B != -1 else self.N
        self.m = min(m if m != -1 else self.B, self.N)  # can't sample more unique items than exist
        self.G = G
        self.k = self.B // self.m
        self.B_i = self.B // self.G
        self.b = min(b_max, self.B_i)
        self.K = -(-self.B_i // self.b)  # ceiling division

        assert self.B % self.m == 0, f"B ({self.B}) must be divisible by m ({self.m})"
        assert self.B % self.G == 0, f"B ({self.B}) must be divisible by number of GPUs ({self.G})"

        self.subsample(0)

    # Each epoch: sample m unique items, repeat each k times → B total indices
    def subsample(self, epoch: int):
        rng = random.Random(self.base_seed + epoch)
        chosen = sorted(rng.sample(range(self.N), self.m))      # pick m
        repeated = [i for idx in chosen for i in [idx]*self.k]  # repeat each k
        self.subsample_indices = repeated

    def __len__(self): return self.B
    def __getitem__(self, i): return self.subsample_indices[i]
    def indices_to_data(self, indices): return [self.all_data[i] for i in indices]


# ---------------------------------------------------------------------------
# LoRA + optimizer mixin (adapted from finetune-stable-diffusion for a CausalLM)
# ---------------------------------------------------------------------------

class LoraMixin:
    """Applies a LoRA adapter to self.model and creates self.optimizer (AdamW).

    Call setup_lora_and_optimizer() during __init__ after self.model exists and is on the right device.
    The frozen base (LoRA adapters disabled) doubles as the GRPO reference policy -- no second model
    copy is needed -- via `accelerator.unwrap_model(self.model).disable_adapter()`.

    Sets:
      self.lora_config                — LoraConfig
      self.trainable_parameters       — filtered list of requires_grad params
      self.optimizer                  — AdamW over trainable parameters
    """

    def setup_lora_and_optimizer(self):
        self.lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
        )
        self.model = get_peft_model(self.model, self.lora_config)

        if self.config.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()  # let checkpointing reach the LoRA params through the frozen base

        self.trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(self.trainable_parameters, lr=self.config.train.learning_rate)

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.model.config = self.accelerator.unwrap_model(self.model).config  # let callers read the model config through the wrapper
