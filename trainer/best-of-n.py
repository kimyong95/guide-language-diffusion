"""Best-of-N baseline -- one greedy chain, no search and no model update.

    for each epoch:
        resps = [generate(prompt(ref_data=data.ref_data)) for _ in range(N)]
        if max(evaluate(resps)) > data.reward: data = argmax(resps)

The buffer is a single (ref_data, reward): the best among every sample ever drawn, so each epoch
re-prompts N times with that one winner. The ratchet runs on the gathered pairs, which are
identical on every rank, so all ranks agree on it without extra communication. m = 1, so the
dataloader hands out the same single task item every time and sampling_step never re-subsamples.
"""

import math
import sys

import torch
from absl import flags
from accelerate.utils import gather_object
from ml_collections import config_flags
from torch.utils.data import DataLoader
from tqdm import tqdm

from base import BaseTrainer
from mixins import DistributedSubsampleDataset
from utils import concat

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/best-of-n.py", "Training configuration.")


class Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

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

        self.data = {"ref_data": None, "reward": -math.inf}

    def run(self):
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        cfg = self.config.sample

        training_data = []
        for data_ids in tqdm(self.training_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            prompts = [self.task.prompt(int(data_id), self.data["ref_data"]) for data_id in data_ids]
            prompt_tokens = self.pipeline.texts_to_tokens(prompts, system_prompt=self.task.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking)   # 2D list (N_local_batch, Lp)

            generated_tokens = self.pipeline.generate(prompt_tokens, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)             # 2D list (N_local_batch, Lg)
            generated_texts = self.pipeline.tokens_to_texts(generated_tokens)
            rewards = torch.tensor([self.task.evaluate(int(data_id), text) for data_id, text in zip(data_ids, generated_texts)], device=self.accelerator.device, dtype=torch.float32)

            training_data.append({
                "generated_texts": generated_texts,   # N_local_batch x str
                "rewards": rewards,                   # (N_local_batch,)
            })

        training_data = {key: concat([batch[key] for batch in training_data]) for key in training_data[0]}

        gathered_rewards = self.accelerator.gather(training_data["rewards"])
        gathered_texts = gather_object(training_data["generated_texts"])
        self.update_data(gathered_texts, gathered_rewards)

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_texts, stage="sampling")

    def update_data(self, texts, rewards):
        """
        Args:
            texts: list (N) of str
            rewards: (N,)
        """
        best = int(rewards.argmax())
        if rewards[best].item() > self.data["reward"]:
            self.data = {"ref_data": texts[best], "reward": rewards[best].item()}


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
