import random

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset


class DistributedSubsampleDataset(Dataset):

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
        self.m = min(m if m != -1 else self.B, self.N)
        self.G = G
        self.k = self.B // self.m
        self.B_i = self.B // self.G
        self.b = min(b_max, self.B_i)
        self.K = -(-self.B_i // self.b)

        assert self.B % self.m == 0, f"B ({self.B}) must be divisible by m ({self.m})"
        assert self.B % self.G == 0, f"B ({self.B}) must be divisible by number of GPUs ({self.G})"

        self.subsample(0)

    def subsample(self, epoch: int):
        rng = random.Random(self.base_seed + epoch)
        chosen = sorted(rng.sample(range(self.N), self.m))
        repeated = [i for idx in chosen for i in [idx]*self.k]
        self.subsample_indices = repeated

    def __len__(self): return self.B
    def __getitem__(self, i): return self.subsample_indices[i]
    def indices_to_data(self, indices): return [self.all_data[i] for i in indices]


class LoraMixin:

    def setup_lora_and_optimizer(self):
        self.lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
        )
        self.model = get_peft_model(self.model, self.lora_config)

        if self.config.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()

        self.trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(self.trainable_parameters, lr=self.config.train.learning_rate)

        # Use self.model_ddp to sync gradients across processes
        self.model_ddp, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)