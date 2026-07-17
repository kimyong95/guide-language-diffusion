import os
import sys

import torch
from absl import flags
from accelerate.utils import gather_object
from ml_collections import config_flags
from tqdm import tqdm

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/best-of-n.py", "Training configuration.")


class Trainer(BaseTrainer):
    """Best-of-N baseline (no search): each epoch draws N independent samples and evaluates each.
    The task ratchets its own best code across processes, so every epoch re-prompts with the best
    code found so far -- but there is no archive, no islands, no guidance. The epoch / N-per-epoch
    loop mirrors optimize-noise; greedy AR generation via model.generate."""

    def __init__(self, config):
        super().__init__(config)

        N = self.config.sample.total_samples
        G = self.accelerator.num_processes
        assert N % G == 0, "total_samples must be divisible by num_processes"
        self.N_local = N // G

        self.token_records = []  # accumulated {epoch, idx, token_ids (long), reward (float)}, saved each epoch

    @torch.no_grad()
    def generate(self, system_prompt, user_prompt):
        # Greedy AR decode of the whole response in one model.generate call.
        prompt_tokens = self.build_prompt_tokens(user_prompt, system_prompt=system_prompt, enable_thinking=self.config.sample.enable_thinking)
        out = self.model.generate(prompt_tokens, max_new_tokens=self.config.sample.max_new_tokens, do_sample=False)
        gen_tokens = out[0, prompt_tokens.shape[1]:]  # strip the echoed prompt
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True), gen_tokens

    def run(self):
        self.model.eval()
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        system_prompt, user_prompt = self.task.prompt()  # asks for a rewrite of the best code so far

        responses = []
        rewards = []
        token_ids = []
        for _ in range(self.N_local):
            response, gen_tokens = self.generate(system_prompt, user_prompt)
            rewards.append(self.task.evaluate(response))  # ratchets the task's best code across ranks
            responses.append(response)
            token_ids.append(gen_tokens.cpu())        # variable-length long tensor per sample

        rewards = torch.tensor(rewards, device=self.accelerator.device, dtype=torch.float32)

        gathered_responses = gather_object(responses)         # variable-length strings -> object gather
        gathered_token_ids = gather_object(token_ids)         # variable-length tensors -> object gather
        gathered_rewards = self.accelerator.gather(rewards)

        self.save_token_ids(epoch, gathered_token_ids, gathered_rewards)

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_responses, stage="sampling")

    def save_token_ids(self, epoch, token_ids, rewards):
        # Persist every generated sample's token ids (long) with its reward (float). The full record
        # list is re-saved each epoch so the file always holds the complete run.
        if not self.accelerator.is_main_process:
            return
        for idx, (ids, reward) in enumerate(zip(token_ids, rewards)):
            self.token_records.append({"epoch": epoch, "idx": idx, "token_ids": ids.cpu().long(), "reward": reward.item()})
        wandb_dir = self.accelerator.get_tracker("wandb").run.dir.removesuffix("/files")
        torch.save(self.token_records, os.path.join(wandb_dir, "generations.pt"))


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()