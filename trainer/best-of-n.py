import math
import sys

import torch
from absl import flags
from accelerate.utils import gather_object
from ml_collections import config_flags
from tqdm import tqdm

import problems
from base import BaseTrainer
from utils import batch_slices, concat

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/best-of-n.py", "Training configuration.")


class Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        assert config.sample.total_samples % self.accelerator.num_processes == 0, f"total_samples ({config.sample.total_samples}) must be divisible by number of GPUs ({self.accelerator.num_processes})"

        self.N_local = config.sample.total_samples // self.accelerator.num_processes

        self.data = {"ref_data": None, "reward": -math.inf}

    def setup_task(self):
        self.problem = problems.get_problem(self.config.problem)

    def run(self):
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        cfg = self.config.sample

        training_data = []
        for batch in tqdm(list(batch_slices(self.N_local, cfg.max_batch_size_per_device)), desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            prompts = [self.problem.prompt(self.data["ref_data"])] * (batch.stop - batch.start)
            prompt_tokens = self.pipeline.texts_to_tokens(prompts, system_prompt=self.problem.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking)   # 2D list (N_local_batch, Lp)

            generated_tokens = self.pipeline.generate(prompt_tokens, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)                # 2D list (N_local_batch, Lg)
            generated_texts = self.pipeline.tokens_to_texts(generated_tokens)
            rewards = torch.tensor([self.problem.evaluate(text) for text in generated_texts], device=self.accelerator.device, dtype=torch.float32)

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
