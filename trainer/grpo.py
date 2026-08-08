import sys

import torch
from absl import flags
from accelerate.utils import gather_object
from ml_collections import config_flags
from torch.utils.data import DataLoader
from tqdm import tqdm

from base import BaseTrainer
from mixins import DistributedSubsampleDataset, LoraMixin
from utils import concat

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/grpo.py", "Training configuration.")


class Trainer(BaseTrainer, LoraMixin):

    def __init__(self, config):
        super().__init__(config)
        self.setup_lora_and_optimizer()

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

        assert self.train_dataset.N_local % config.train.gradient_updates_per_epoch == 0, f"per-rank rollouts N_local ({self.train_dataset.N_local}) must be divisible by gradient_updates_per_epoch ({config.train.gradient_updates_per_epoch})"
        self.accelerator.gradient_accumulation_steps = self.train_dataset.N_local // config.train.gradient_updates_per_epoch

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
        self.pipeline.model.eval()
        cfg = self.config.sample

        self.train_dataset.subsample(epoch)
        training_data = []
        for data_ids in tqdm(self.training_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            prompts = [self.task.prompt(int(data_id)) for data_id in data_ids]
            prompt_tokens = self.pipeline.texts_to_tokens(prompts, system_prompt=self.task.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking)  # 2D list (N_local_batch, Lp)
            generated_tokens = self.pipeline.generate(prompt_tokens, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)            # 2D list (N_local_batch, Lg)
            generated_texts = self.pipeline.tokens_to_texts(generated_tokens)
            rewards = torch.tensor([self.task.evaluate(int(data_id), text) for data_id, text in zip(data_ids, generated_texts)], device=self.accelerator.device, dtype=torch.float32)
            entropies = torch.tensor([self.pipeline.entropy(prompt, generated).mean().item() for prompt, generated in zip(prompt_tokens, generated_tokens)], device=self.accelerator.device, dtype=torch.float32)  # nats/token: mean over Lg

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
        training_data["advantages"] = self.ungather(gathered_advantages)   # (N_local,)

        group_reward_std = torch.stack([gathered_rewards[[i for i, x in enumerate(gathered_data_ids) if x == data_id]].std() for data_id in set(gathered_data_ids)]).mean()

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling", extra={"sampling/reward-group-std": group_reward_std.item(), "sampling/entropy": gathered_entropy.item()})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_texts, stage="sampling")

        return training_data

    def training_step(self, epoch, training_data):
        self.pipeline.model.train()
        cfg = self.config.train
        beta, clip_range = cfg.beta, cfg.clip_range

        prompt_tokens_list, generated_tokens_list = training_data["prompt_tokens"], training_data["generated_tokens"]
        advantages = training_data["advantages"]

        with torch.no_grad():
            old_log_probs_list = [self.pipeline.log_probs(prompt_tokens, generated_tokens) for prompt_tokens, generated_tokens in zip(prompt_tokens_list, generated_tokens_list)]

        losses, kls, grad_norm = [], [], torch.tensor(0.0)
        for prompt_tokens, generated_tokens, old_log_probs, advantage in zip(prompt_tokens_list, generated_tokens_list, old_log_probs_list, advantages):
            with self.accelerator.accumulate(self.pipeline.model):
                log_probs = self.pipeline.log_probs(prompt_tokens, generated_tokens)   # (Lg,), through the adapter
                with self.accelerator.unwrap_model(self.pipeline.model).disable_adapter(), torch.no_grad():
                    ref_log_probs = self.pipeline.log_probs(prompt_tokens, generated_tokens)   # same weights, adapter off

                ref_log_ratio = torch.clamp(ref_log_probs - log_probs, min=-20, max=20)
                kl = torch.clamp(torch.exp(ref_log_ratio) - ref_log_ratio - 1, min=-10, max=10)
                ratio = torch.exp(torch.clamp(log_probs - old_log_probs, min=-20, max=20))
                clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                ppo_loss = torch.min(ratio * advantage, clipped_ratio * advantage)
                ppo_loss = -(ppo_loss - beta * kl)
                loss = ppo_loss.mean()

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.pipeline.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                losses.append(loss.detach())
                kls.append(kl.mean().detach())

        loss_value = torch.stack(losses).mean().reshape(1)
        kl_value = torch.stack(kls).mean().reshape(1)
        objective_evaluations = epoch * self.config.sample.total_samples
        self.accelerator.log({
            "objective-evaluations": objective_evaluations,
            "training/loss": self.accelerator.gather(loss_value).mean().item(),
            "training/kl": self.accelerator.gather(kl_value).mean().item(),
            "training/grad-norm": grad_norm.item(),
        })


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
