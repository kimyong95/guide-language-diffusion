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
from utils import clamp_preserve_grad, concat, iter_dict

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/optimize-hidden.py", "Training configuration.")


class Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        if config.train.gradient_checkpointing:
            # non-reentrant: x is written in by a layer hook, not passed in as a checkpointed input
            self.pipeline.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        D = self.pipeline.config.hidden_size
        gen = torch.Generator().manual_seed(config.seed)
        x = self.project_to_sphere(torch.randn(config.model.n_intervene, D, generator=gen))
        self.x = x.to(self.accelerator.device, torch.float32).requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.x], lr=config.train.learning_rate)

        self.train_dataset = DistributedSubsampleDataset(all_data=self.sample_task.data,N=config.sample.total_samples,G=self.accelerator.num_processes,N_batch_max=config.model.max_batch_size_per_device,m=config.sample.m,base_seed=config.seed,)
        training_dataloader = DataLoader(self.train_dataset, batch_size=self.train_dataset.N_local_batch, shuffle=False)
        self.training_dataloader = self.accelerator.prepare(training_dataloader)

        self.val_dataset = DistributedSubsampleDataset(all_data=self.val_task.data,N=-1,G=self.accelerator.num_processes,N_batch_max=config.model.max_batch_size_per_device,k=config.val.k,base_seed=config.seed,)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.val_dataset.N_local_batch, shuffle=False)
        self.val_dataloader = self.accelerator.prepare(val_dataloader)

    @staticmethod
    def project_to_sphere(x):
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True) * math.sqrt(x.shape[-1])

    @staticmethod
    def compute_reward_statistics(data_ids, rewards):
        means, stds = torch.zeros_like(rewards), torch.zeros_like(rewards)
        for data_id in set(data_ids):
            indices = [i for i, x in enumerate(data_ids) if x == data_id]
            means[indices], stds[indices] = rewards[indices].mean(), rewards[indices].std()
        return means, stds

    def run(self):
        self.validation_step(epoch=0)
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            training_data = self.sampling_step(epoch)
            self.training_step(epoch=epoch, training_data=training_data)
            if epoch % self.config.val.every_n_epochs == 0:
                self.validation_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.pipeline.model.eval()
        cfg = self.config.model

        self.train_dataset.subsample(epoch)
        training_data = []
        for data_ids in tqdm(self.training_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            prompts = [self.sample_task.prompt(int(data_id)) for data_id in data_ids]
            prompt_tokens = self.pipeline.texts_to_tokens(prompts, system_prompt=self.sample_task.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking, n_intervene=cfg.n_intervene)  # 2D list (N_local_batch, Lp)
            x = einops.repeat(self.x.to(self.pipeline.model.dtype), "Lx D -> N Lx D", N=len(prompt_tokens))                                                                                                      # (N_local_batch, Lx, D)
            generated_output = self.pipeline.generate(prompt_tokens, x, max_new_tokens=cfg.max_new_tokens)
            generated_tokens = generated_output.tokens                                                                    # 2D list (N_local_batch, Lg)
            generated_texts = generated_output.texts
            rewards = torch.tensor([self.sample_task.evaluate(int(data_id), text) for data_id, text in zip(data_ids, generated_texts)], device=self.accelerator.device, dtype=torch.float32)
            entropies = generated_output.entropies   # (N_local_batch,) nats/token
            generated_lengths = torch.tensor([len(tokens) for tokens in generated_tokens], device=self.accelerator.device, dtype=torch.float32)

            training_data.append({
                "data_ids": data_ids,                     # (N_local_batch,)
                "prompt_tokens": prompt_tokens,           # 2D list (N_local_batch, Lp), ragged
                "generated_tokens": generated_tokens,     # 2D list (N_local_batch, Lg), ragged
                "generated_texts": generated_texts,       # N_local_batch x str
                "rewards": rewards,                       # (N_local_batch,)
                "entropies": entropies,                   # (N_local_batch,)
                "generated_lengths": generated_lengths,   # (N_local_batch,)
            })

        training_data = {key: concat([batch[key] for batch in training_data]) for key in training_data[0]}

        gathered_data_ids = self.accelerator.gather(training_data["data_ids"]).tolist()
        gathered_rewards = self.accelerator.gather(training_data["rewards"])
        gathered_entropy = self.accelerator.gather(training_data["entropies"]).mean()
        gathered_length = self.accelerator.gather(training_data["generated_lengths"])
        gathered_texts = gather_object(training_data["generated_texts"])
        gathered_means, _ = self.compute_reward_statistics(gathered_data_ids, gathered_rewards)
        training_data["reward_means"] = self.ungather(gathered_means)   # (N_local,)
        training_data["mean_generated_lengths"] = gathered_length.mean().expand_as(training_data["rewards"])

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling", extra={"sampling/entropy": gathered_entropy.item()})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_texts, stage="sampling")

        return training_data

    @torch.no_grad()
    def validation_step(self, epoch):
        self.pipeline.model.eval()
        cfg = self.config.model

        val_data = []
        for data_ids in tqdm(self.val_dataloader, desc="Validation", position=1, leave=False, disable=not self.accelerator.is_main_process):
            prompts = [self.val_task.prompt(int(data_id)) for data_id in data_ids]
            prompt_tokens = self.pipeline.texts_to_tokens(prompts, system_prompt=self.val_task.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking, n_intervene=cfg.n_intervene)  # 2D list (N_local_batch, Lp)
            x = einops.repeat(self.x.to(self.pipeline.model.dtype), "Lx D -> N Lx D", N=len(prompt_tokens))   # (N_local_batch, Lx, D)
            generated_output = self.pipeline.generate(prompt_tokens, x, max_new_tokens=cfg.max_new_tokens)
            generated_texts = generated_output.texts
            rewards = torch.tensor([self.val_task.evaluate(int(data_id), text) for data_id, text in zip(data_ids, generated_texts)], device=self.accelerator.device, dtype=torch.float32)
            entropies = generated_output.entropies   # (N_local_batch,) nats/token

            val_data.append({
                "data_ids": data_ids,                 # (N_local_batch,)
                "generated_texts": generated_texts,   # N_local_batch x str
                "rewards": rewards,                   # (N_local_batch,)
                "entropies": entropies,               # (N_local_batch,)
            })

        val_data = {key: concat([batch[key] for batch in val_data]) for key in val_data[0]}

        gathered_data_ids = self.accelerator.gather(val_data["data_ids"]).tolist()
        gathered_rewards = self.accelerator.gather(val_data["rewards"])
        gathered_entropy = self.accelerator.gather(val_data["entropies"]).mean()
        gathered_texts = gather_object(val_data["generated_texts"])
        pass_at_k = torch.stack([gathered_rewards[[i for i, x in enumerate(gathered_data_ids) if x == data_id]].max() for data_id in set(gathered_data_ids)]).mean()

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="validation", extra={"validation/pass-at-k": pass_at_k.item(), "validation/entropy": gathered_entropy.item()})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_texts, stage="validation")

    def training_step(self, epoch, training_data):
        self.pipeline.model.train()   # for gradient checkpointing
        cfg = self.config.train
        N_local = len(training_data["rewards"])

        self.optimizer.zero_grad()
        losses = []
        with self.pipeline.unpaged():
            for sample in iter_dict(training_data):
                prompt_tokens, generated_tokens = sample["prompt_tokens"], sample["generated_tokens"]
                reward, reward_mean = sample["rewards"], sample["reward_means"]
                mean_generated_length = sample["mean_generated_lengths"]

                with self.pipeline.intervene(self.x[None].to(self.pipeline.model.dtype), [prompt_tokens]) as [intervene_prompt_tokens]:
                    cur_log_probs = self.pipeline.log_probs(intervene_prompt_tokens, generated_tokens)   # (Lg,)
                    log_ratio = cur_log_probs - cur_log_probs.detach()                         # (Lg,), zero-valued: x takes one update per epoch
                    negative_ratio = (1 - reward_mean * torch.exp(log_ratio)) * torch.nan_to_num(1 / (1 - reward_mean), posinf=0.0) # prevent nan when all rewards are 1.0
                    negative_log_ratio = torch.log(clamp_preserve_grad(negative_ratio, cfg.epsilon))
                    nft_loss = - (1 - reward_mean) * (reward * log_ratio + (1 - reward) * negative_log_ratio)   # (Lg,)
                    loss = nft_loss.sum() / mean_generated_length
                    self.accelerator.backward(loss / N_local)
                losses.append(loss.detach())

        self.x.grad = self.accelerator.reduce(self.x.grad, reduction="mean")
        grad_norm = self.x.grad.norm(dim=-1).mean()
        self.optimizer.step()
        with torch.no_grad():
            self.x.copy_(self.project_to_sphere(self.x))
        self.x.data = broadcast(self.x.data)

        loss_value = torch.stack(losses).mean()
        gathered_loss = self.accelerator.gather(loss_value.reshape(1)).mean().item()
        objective_evaluations = epoch * self.config.sample.total_samples
        self.accelerator.log({
            "objective-evaluations": objective_evaluations,
            "training/loss": gathered_loss,
            "training/grad-norm": grad_norm.item(),
        })


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()