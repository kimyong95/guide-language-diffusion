import sys

import einops
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


def completion_log_probs(model, context_tokens, generated_tokens):
    """Per-token log-probs of `generated_tokens` continued from `context_tokens`, under `model`.

    Both are 1-D unpadded token rows; their concatenation has no padding, so no attention_mask (all ones)
    and no position_ids (the default arange is correct) are needed."""
    input_tokens = torch.cat([context_tokens, generated_tokens])         # (L,)
    logits = model(input_tokens.unsqueeze(0)).logits[0, :-1, :]           # (L-1, V)
    log_probs = logits.float().log_softmax(dim=-1)
    token_log_probs = log_probs.gather(1, input_tokens[1:, None]).squeeze(1)   # (L-1,)
    return token_log_probs[context_tokens.shape[0] - 1:]                  # generated tokens only


class Trainer(BaseTrainer, LoraMixin):
    """Token-level GRPO with LoRA on GSM8K. Each epoch subsamples m questions (each repeated k = B/m
    times), draws them as flat b-sized batches, generates one completion per rollout, then
    z-score-normalizes reward within each question's group (across the gathered global batch) into
    advantages and takes a PPO-clipped policy-gradient step with a KL penalty to the frozen base
    (= LoRA adapters disabled). Padding lives only inside the batched generate(); every log-prob forward
    runs on a single unpadded sequence."""

    def __init__(self, config):
        super().__init__(config)
        self.setup_lora_and_optimizer()

        # b = min(max_batch_size_per_device, B/G) is computed inside the dataset; the dataloader batches at
        # that size and accelerate.prepare shards the flat rollouts across ranks.
        self.train_dataset = DistributedSubsampleDataset(
            all_data=self.task.data,
            B=config.sample.total_samples,
            G=self.accelerator.num_processes,
            m=config.sample.m,
            b_max=config.sample.max_batch_size_per_device,
            base_seed=config.seed,
        )
        training_dataloader = DataLoader(self.train_dataset, batch_size=self.train_dataset.b, shuffle=True)
        self.training_dataloader = self.accelerator.prepare(training_dataloader)

        # One micro-step per rollout (per-sequence forward), B_i rollouts per rank per epoch. Fix the
        # gradient-accumulation window so each epoch makes exactly gradient_updates_per_epoch optimizer steps.
        assert self.train_dataset.B_i % config.train.gradient_updates_per_epoch == 0, f"per-rank rollouts B_i ({self.train_dataset.B_i}) must be divisible by gradient_updates_per_epoch ({config.train.gradient_updates_per_epoch})"
        self.accelerator.gradient_accumulation_steps = self.train_dataset.B_i // config.train.gradient_updates_per_epoch

    @staticmethod
    def compute_advantages(data_ids, rewards):
        """Group-normalized advantages: within each question's group, z-score the reward with
        (reward - mean) / (std + 1e-4). Grouping is by data id, over the gathered global batch."""
        advantages = torch.zeros_like(rewards)
        for data_id in set(data_ids):
            indices = [i for i, x in enumerate(data_ids) if x == data_id]
            group = rewards[indices]
            advantages[indices] = (group - group.mean()) / (group.std() + 1e-4)
        return advantages

    def run(self):
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            training_data = self.sampling_step(epoch)
            self.training_step(epoch=epoch, training_data=training_data)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.model.eval()
        model = self.accelerator.unwrap_model(self.model)  # generate() lives on the (unwrapped) peft model
        cfg = self.config.sample

        self.train_dataset.subsample(epoch)
        training_data = []      # one dict per b-sized batch; merged with concat() below
        for data_ids in tqdm(self.training_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            # apply_chat_template tokenizes AND left-pads the whole batch in one call, so every prompt
            # shares a single prompt_length P (their right edges align).
            prompt_texts = [
                [{"role": "system", "content": self.task.SYSTEM_PROMPT}, {"role": "user", "content": self.task.prompt(int(data_id))}]
                for data_id in data_ids
            ]
            prompt_tokens_data = self.tokenizer.apply_chat_template(
                prompt_texts, tokenize=True, padding=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=cfg.enable_thinking,
            ).to(self.accelerator.device)
            prompt_tokens, prompt_attention = prompt_tokens_data.input_ids, prompt_tokens_data.attention_mask  # (b, P), left-padded
            prompt_length = prompt_tokens.shape[1]

            generated_tokens = model.generate(
                prompt_tokens,
                attention_mask=prompt_attention,
                do_sample=True,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )  # (b, P + T), completions right-padded with pad_token_id

            # Unpad each rollout: drop the prompt's left-pad, cut the completion after its first EOS.
            prompt_tokens = self.strip_pads(prompt_tokens)                          # b x (P_i,)
            generated_tokens = self.strip_eos(generated_tokens[:, prompt_length:])  # b x (T_i,)
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            rewards = torch.tensor([self.task.evaluate(int(data_id), text) for data_id, text in zip(data_ids, generated_texts)],
                                   device=self.accelerator.device, dtype=torch.float32)

            training_data.append({
                "data_ids": data_ids,                 # (b,)
                "prompt_tokens": prompt_tokens,       # b x (P_i,)
                "generated_tokens": generated_tokens, # b x (T_i,)
                "generated_texts": generated_texts,   # b x str
                "rewards": rewards,                   # (b,)
            })

        # Flatten the per-batch dicts into one dict of (tensor | list) over this rank's B_i rollouts.
        training_data = {key: concat([batch[key] for batch in training_data]) for key in training_data[0]}

        # Advantages on the gathered global batch, then scattered back to this rank's rollouts.
        gathered_data_ids = self.accelerator.gather(training_data["data_ids"]).tolist()
        gathered_rewards = self.accelerator.gather(training_data["rewards"])
        gathered_texts = gather_object(training_data["generated_texts"])
        gathered_advantages = self.compute_advantages(gathered_data_ids, gathered_rewards)
        training_data["advantages"] = einops.rearrange(gathered_advantages, "(process batch) -> process batch", process=self.accelerator.num_processes)[self.accelerator.process_index]

        # Average std within each group, near 0 means no learning signal
        group_reward_std = torch.stack([gathered_rewards[[i for i, x in enumerate(gathered_data_ids) if x == data_id]].std() for data_id in set(gathered_data_ids)]).mean()

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling", extra={"sampling/reward-group-std": group_reward_std.item()})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_texts, stage="sampling")

        return training_data

    def training_step(self, epoch, training_data):
        self.model.train()
        cfg = self.config.train
        beta, clip_range = cfg.beta, cfg.clip_range

        prompt_tokens_list, generated_tokens_list = training_data["prompt_tokens"], training_data["generated_tokens"]
        advantages = training_data["advantages"]

        # Behavior-policy snapshot: old log-probs of every rollout, taken before any optimizer step (the
        # fixed PPO reference; equals the policy that generated the rollouts this epoch).
        with torch.no_grad():
            old_log_probs_list = [completion_log_probs(self.model, prompt_tokens, generated_tokens) for prompt_tokens, generated_tokens in zip(prompt_tokens_list, generated_tokens_list)]

        losses, kls, grad_norm = [], [], torch.tensor(0.0)
        for prompt_tokens, generated_tokens, old_log_probs, advantage in zip(prompt_tokens_list, generated_tokens_list, old_log_probs_list, advantages):
            with self.accelerator.accumulate(self.model):
                # Current policy (grad, LoRA enabled).
                log_probs = completion_log_probs(self.model, prompt_tokens, generated_tokens)
                # Reference policy = frozen base (LoRA disabled) via disable_adapter() on the peft model.
                with self.accelerator.unwrap_model(self.model).disable_adapter(), torch.no_grad():
                    ref_log_probs = completion_log_probs(self.model, prompt_tokens, generated_tokens)

                # GRPO loss; advantage is the group scalar.
                per_token_kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                per_token_loss = torch.min(ratio * advantage, clipped_ratio * advantage)
                per_token_loss = -(per_token_loss - beta * per_token_kl)
                loss = per_token_loss.mean()

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                losses.append(loss.detach())
                kls.append(per_token_kl.mean().detach())

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
