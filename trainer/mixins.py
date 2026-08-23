import random

import torch
from accelerate.utils import broadcast
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset


class DistributedSubsampleDataset(Dataset):

    def __init__(self, all_data, N, G, N_batch_max, m=None, k=None, base_seed=0):
        # N_all         : total number of items
        # N             : total samples per epoch (across all GPUs)   (N == -1 → one pass over the dataset, N = N_all*k)
        # m             : number of unique items sampled per epoch    (give exactly one of m, k; N = m*k fixes the other)
        # k             : repetitions per item per epoch
        # G             : number of GPUs (processes)
        # N_local       : total samples per epoch per GPU             (N_local = N/G)
        # N_batch_max   : max batch size per GPU
        # N_local_batch : actual batch size per GPU                   (N_local_batch = min(N_batch_max, N/G))
        # K             : number of batches per epoch per GPU         (K = N_local//N_local_batch)

        assert (m is None) != (k is None), "give exactly one of m and k; N = m*k fixes the other"
        assert N != -1 or k is not None, "one pass over the dataset is sized by k, not m"

        self.all_data = all_data
        self.N_all = len(self.all_data)
        self.base_seed = base_seed
        self.N = N if N != -1 else self.N_all * k
        self.m = m if m is not None else self.N // k
        self.k = k if k is not None else self.N // self.m
        self.G = G
        self.N_local = self.N // self.G
        self.N_local_batch = min(N_batch_max, self.N_local)
        self.K = -(-self.N_local // self.N_local_batch)

        assert self.m * self.k == self.N, f"N ({self.N}) must equal m ({self.m}) * k ({self.k})"
        assert self.m <= self.N_all, f"m ({self.m}) must not exceed the dataset size ({self.N_all})"
        assert self.N % self.G == 0, f"N ({self.N}) must be divisible by number of GPUs ({self.G})"

        self.subsample(0)

    def subsample(self, epoch: int):
        rng = random.Random(self.base_seed + epoch)
        chosen = sorted(rng.sample(range(self.N_all), self.m))
        repeated = [i for idx in chosen for i in [idx]*self.k]
        self.subsample_indices = repeated

    def __len__(self): return self.N
    def __getitem__(self, i): return self.subsample_indices[i]
    def indices_to_data(self, indices): return [self.all_data[i] for i in indices]


class LoraMixin:

    def setup_lora_and_optimizer(self):
        self.lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
        )
        self.pipeline.model = get_peft_model(self.pipeline.model, self.lora_config)

        if self.config.train.gradient_checkpointing:
            self.pipeline.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.pipeline.model.enable_input_require_grads()

        self.trainable_parameters = list(filter(lambda p: p.requires_grad, self.pipeline.model.parameters()))
        broadcast([parameter.data for parameter in self.trainable_parameters])
        self.optimizer = torch.optim.AdamW(self.trainable_parameters, lr=self.config.train.learning_rate)
        self.optimizer = self.accelerator.prepare(self.optimizer)