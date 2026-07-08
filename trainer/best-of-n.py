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
    """Best-of-N baseline (no search): each epoch draws N independent samples, evaluates each, logs
    the batch reward, and tracks the best code so far. Every sample is prompted with the current
    best code plus the initial seed -- so the pool ratchets toward the best found, but there is no
    archive, no islands, no guidance. The epoch / N-per-epoch loop mirrors flow-guide; block-diffusion
    generation mirrors open-evolve."""

    def __init__(self, config):
        super().__init__(config)

        N = self.config.sample.total_samples
        G = self.accelerator.num_processes
        assert N % G == 0, "total_samples must be divisible by num_processes"
        self.N_local = N // G

        self.timesteps = self.pipeline.scheduler.set_timesteps(
            num_inference_steps=self.config.sample.num_inference_steps, device=self.accelerator.device
        )

        # Seed the best-so-far with the initial code (evaluated once, deterministic across ranks).
        init_code = self.task.initial_code()
        self.initial = (init_code, self.task.evaluate_code(init_code))  # (code, reward), fixed
        self.best = self.initial                                   # (code, reward), updated per epoch

    @torch.no_grad()
    def generate(self, prompt):
        # Block diffusion: denoise the whole canvas_length canvas per block, argmax-commit it, grow the
        # kv_cache, repeat until EOS or the token budget is exhausted (max_tokens // canvas_length blocks).
        pipeline = self.pipeline
        prompt_tokens = pipeline.build_prompt_tokens(prompt, enable_thinking=self.config.sample.enable_thinking)
        kv_cache = pipeline.build_kv_cache(prompt_tokens)

        generated = []
        for _ in range(self.config.sample.max_tokens // self.config.sample.canvas_length):
            xt_logits = None
            xt_tokens = pipeline.sample_init_tokens()[None]
            for timestep in self.timesteps:
                xt_logits, _, finished = pipeline.model_predict(xt_tokens, xt_logits, timestep, kv_cache)
                xt_tokens = pipeline.sample_logits_to_tokens(xt_logits)[None]
                if finished[-1]:
                    break
            canvas = pipeline.argmax_logits_to_tokens(xt_logits)
            generated.append(canvas)
            if torch.isin(canvas, pipeline.eos_token_id).any():
                break
            kv_cache = pipeline.build_kv_cache(canvas[None], kv_cache)

        gen_tokens = pipeline.strip_thinking_tokens(torch.cat(generated))
        return pipeline.processor.decode(gen_tokens, skip_special_tokens=True)

    def run(self):
        self.pipeline.model.eval()
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        prompt = self.task.build_prompt([self.best, self.initial])  # current best + the initial seed

        codes = []
        rewards = []
        for _ in range(self.N_local):
            response = self.generate(prompt)
            code = self.task.extract_code(response)
            rewards.append(self.task.evaluate_code(code))
            codes.append(code)

        rewards = torch.tensor(rewards, device=self.accelerator.device, dtype=torch.float32)

        gathered_codes = gather_object(codes)                 # variable-length strings -> object gather
        gathered_rewards = self.accelerator.gather(rewards)
        best_idx = gathered_rewards.argmax().item()
        if gathered_rewards[best_idx].item() > self.best[1]:
            self.best = (gathered_codes[best_idx], gathered_rewards[best_idx].item())

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling", extra={"sampling/best-so-far": self.best[1]})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_codes, stage="sampling")


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()