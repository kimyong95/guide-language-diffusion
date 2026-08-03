import math
import sys

import einops
import torch
from absl import flags
from accelerate.utils import broadcast, gather_object
from ml_collections import config_flags
from torch.utils.data import DataLoader
from tqdm import tqdm

from base import BaseTrainer
from mixins import DistributedSubsampleDataset
from utils import concat

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/optimize-hidden.py", "Training configuration.")




class Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        H, D = self.pipeline.config.num_hidden_layers, self.pipeline.config.hidden_size
        gen = torch.Generator().manual_seed(config.seed)
        x = self.project_to_sphere(torch.randn(H, config.sample.n_intervene, D, generator=gen))
        self.x = x.to(self.accelerator.device, torch.float32).requires_grad_(True)  # fp32 master copy
        self.optimizer = torch.optim.Adam([self.x], lr=config.train.learning_rate)

        self.train_dataset = DistributedSubsampleDataset(
            all_data=self.task.data,
            N=config.sample.total_samples,
            G=self.accelerator.num_processes,
            m=config.sample.m,
            N_batch_max=config.sample.max_batch_size_per_device,
            base_seed=config.seed,
        )
        training_dataloader = DataLoader(self.train_dataset, batch_size=self.train_dataset.N_local_batch, shuffle=False)
        self.training_dataloader = self.accelerator.prepare(training_dataloader)

    @staticmethod
    def project_to_sphere(x):
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True) * math.sqrt(x.shape[-1])

    @staticmethod
    def compute_advantages(data_ids, rewards):
        advantages = torch.zeros_like(rewards)
        for data_id in set(data_ids):
            indices = [i for i, x in enumerate(data_ids) if x == data_id]
            group = rewards[indices]
            advantages[indices] = (group - group.mean()) / (group.std() + 1e-6)
        return advantages

    def run(self):
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            training_data = self.sampling_step(epoch)
            self.training_step(epoch=epoch, training_data=training_data)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        cfg = self.config.sample

        self.train_dataset.subsample(epoch)
        training_data = []
        for data_ids in tqdm(self.training_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            prompts = [self.task.prompt(int(data_id)) for data_id in data_ids]
            prompt_tokens = self.pipeline.texts_to_tokens(prompts, system_prompt=self.task.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking, n_intervene=cfg.n_intervene)  # 2D list (N_local_batch, Lp)
            x = einops.repeat(self.x, "h lx d -> n h lx d", n=len(prompt_tokens))                                                                                           # (N_local_batch, H, Lx, D)
            generated_tokens = self.pipeline.generate(prompt_tokens, x, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)                                     # 2D list (N_local_batch, Lg)
            generated_texts = self.pipeline.tokens_to_texts(generated_tokens)
            rewards = torch.tensor([self.task.evaluate(int(data_id), text) for data_id, text in zip(data_ids, generated_texts)], device=self.accelerator.device, dtype=torch.float32)
            entropies = torch.tensor([self.pipeline.entropy(prompt, generated, self.x[None]).mean().item() for prompt, generated in zip(prompt_tokens, generated_tokens)], device=self.accelerator.device, dtype=torch.float32)  # nats/token: mean over Lg

            training_data.append({
                "data_ids": data_ids,                 # (N_local_batch,)
                "prompt_tokens": prompt_tokens,       # 2D list (N_local_batch, Lp), ragged
                "generated_tokens": generated_tokens, # 2D list (N_local_batch, Lg), ragged
                "generated_texts": generated_texts,   # N_local_batch x str
                "rewards": rewards,                   # (N_local_batch,)
                "entropies": entropies,               # (N_local_batch,)
            })

        training_data = {key: concat([batch[key] for batch in training_data]) for key in training_data[0]}

        gathered_data_ids = self.accelerator.gather(training_data["data_ids"]).tolist()
        gathered_rewards = self.accelerator.gather(training_data["rewards"])
        gathered_entropy = self.accelerator.gather(training_data["entropies"]).mean()
        gathered_texts = gather_object(training_data["generated_texts"])
        gathered_advantages = self.compute_advantages(gathered_data_ids, gathered_rewards)
        training_data["advantages"] = einops.rearrange(gathered_advantages, "(process batch) -> process batch", process=self.accelerator.num_processes)[self.accelerator.process_index]

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling", extra={"sampling/entropy": gathered_entropy.item()})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_texts, stage="sampling")

        return training_data

    def training_step(self, epoch, training_data):
        prompt_tokens_list, generated_tokens_list = training_data["prompt_tokens"], training_data["generated_tokens"]
        advantages = training_data["advantages"]
        N_local = len(advantages)

        self.optimizer.zero_grad()
        losses = []
        for prompt_tokens, generated_tokens, advantage in zip(prompt_tokens_list, generated_tokens_list, advantages):
            log_probs = self.pipeline.log_probs(prompt_tokens, generated_tokens, self.x[None])  # (Lg,)
            loss = -(advantage * log_probs.mean())                                        # length-normalized
            self.accelerator.backward(loss / N_local)  # accumulate; graph freed after each backward
            losses.append(loss.detach())

        self.x.grad = self.accelerator.reduce(self.x.grad, reduction="mean")
        grad_norm = self.x.grad.norm(dim=-1).mean()
        self.optimizer.step()
        with torch.no_grad():
            self.x.copy_(self.project_to_sphere(self.x))
        self.x.data = broadcast(self.x.data)  # keep x bit-identical across ranks

        with torch.no_grad():
            gram = self.x @ self.x.mT / self.x.shape[-1]  # (H, Lx, Lx) cosines: the rows are on the sqrt(D) sphere
            Lx = gram.shape[-1]
            mean_cosine = (gram.sum(dim=(-2, -1)) - Lx) / (Lx * (Lx - 1))  # (H,), the diagonal is exactly 1
            diversity = ((1 - mean_cosine) * (Lx - 1) / Lx).mean()

        loss_value = torch.stack(losses).mean()
        gathered_loss = self.accelerator.gather(loss_value.reshape(1)).mean().item()
        objective_evaluations = epoch * self.config.sample.total_samples
        self.accelerator.log({
            "objective-evaluations": objective_evaluations,
            "training/loss": gathered_loss,
            "training/grad-norm": grad_norm.item(),
            "training/x-diversity": diversity.item(),
        })


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()