import random

import torch
from accelerate.utils import broadcast
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset


class SubsampleDataset(Dataset):

    def __init__(self, all_data, N, m=None, k=None, base_seed=0):
        # N_all : items in all_data                     (N == -1 → one pass over the dataset, N = N_all*k)
        # N     : items in data, one epoch across all GPUs
        # m     : unique items sampled per epoch        (give exactly one of m, k; N = m*k fixes the other)
        # k     : repetitions per item per epoch

        assert (m is None) != (k is None), "give exactly one of m and k; N = m*k fixes the other"
        assert N != -1 or k is not None, "one pass over the dataset is sized by k, not m"

        self.all_data = all_data
        self.N_all = len(self.all_data)
        self.base_seed = base_seed
        self.N = N if N != -1 else self.N_all * k
        self.m = m if m is not None else self.N // k
        self.k = k if k is not None else self.N // self.m

        assert self.m * self.k == self.N, f"N ({self.N}) must equal m ({self.m}) * k ({self.k})"
        assert self.m <= self.N_all, f"m ({self.m}) must not exceed the dataset size ({self.N_all})"

        self.subsample(0)

    def subsample(self, epoch: int):
        rng = random.Random(self.base_seed + epoch)
        chosen = sorted(rng.sample(range(self.N_all), self.m))
        self.data = [self.all_data[idx] for idx in chosen for _ in range(self.k)]

    def __len__(self): return self.N
    def __getitem__(self, i): return self.data[i]


class DistributedDataloader(DataLoader):
    """Iterates only this process's block of the dataset: the contiguous run between the
    i/num_processes and (i+1)/num_processes marks. Sizes differ by at most one -- 5 items over 4
    processes is [1,1,1,2] -- and the blocks tile the dataset with nothing duplicated and nothing
    dropped, since one block's end is the next one's start by construction.

    Contiguous rather than strided, so neighbours in the dataset stay together: the k copies
    SubsampleDataset lays down adjacently land on one process unless a block boundary happens to
    fall inside them, and there are only num_processes-1 boundaries to fall.

    It shards by itself, so it must never be passed through accelerator.prepare -- that would
    shard the shard, and prepare's even_batches padding would duplicate samples on top.

    Blocks differ in size, so the number of batches differs across processes. That is safe only
    while no collective runs inside the loop.
    """

    def __init__(self, dataset, num_processes, process_index, batch_size, **kwargs):
        """
        Args:
            dataset: map-style dataset, given whole and sharded here
            num_processes (int): processes to split it across
            process_index (int): this process
            batch_size (int): upper bound, capped at this process's block
        """
        N = len(dataset)
        assert N >= num_processes, f"N ({N}) must be >= processes ({num_processes}), otherwise some process would get nothing"
        sampler = range(process_index * N // num_processes, (process_index + 1) * N // num_processes)
        super().__init__(dataset, batch_size=min(batch_size, len(sampler)), sampler=sampler, **kwargs)


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