import random
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from absl import flags
from ml_collections import config_flags
from tqdm import tqdm

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/open-evolve.py", "Training configuration.")


@dataclass
class Program:
    code: str
    reward: float
    metrics: dict
    artifacts: dict
    generation: int


class Trainer(BaseTrainer):
    """OpenEvolve-style search: pick a parent program from a flat archive, ask the model to
    full-rewrite it, evaluate, and add the candidate back. The gradient guidance of flow-guide
    is replaced by this evolutionary loop; sampling/logging mirror flow-guide."""

    def __init__(self, config):
        super().__init__(config)

        N = self.config.sample.total_samples
        G = self.dp_size
        assert N % G == 0, "total_samples must be divisible by dp_size"
        self.N_local = N // G

        self.timesteps = self.pipeline.scheduler.set_timesteps(
            num_inference_steps=self.config.sample.num_inference_steps, device=self.accelerator.device
        )

        # Seed the archive with the initial program. Broadcast its reward across the whole world so
        # every rank's archive is identical (a precondition for TP-consistent prompts).
        code0 = self.task.initial_program()
        r0, _, _ = self.task.evaluate_program(code0)
        r0_t = torch.tensor([r0], device=self.accelerator.device)
        dist.broadcast(r0_t, src=0)
        r0 = float(r0_t.item())
        self.database = [Program(code0, r0, {"combined_score": r0}, {}, generation=0)]
        self.best = self.database[0]

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """Block diffusion: denoise the whole canvas per block, argmax-commit it, repeat.
        Copied from eval-gemma.py's DiffusionGemmaBlock.generate, with broadcast_tp added to keep
        tensor-parallel ranks in lockstep (as flow-guide's sampling loop does)."""
        pipeline = self.pipeline
        prompt_tokens = pipeline.build_prompt_tokens(prompt)
        kv_cache = pipeline.build_kv_cache(prompt_tokens)

        generated = []
        for _ in range(self.config.sample.max_blocks):
            xt_logits = None
            xt_tokens = pipeline.sample_init_tokens()[None]
            self.broadcast_tp(xt_tokens)
            for timestep in self.timesteps:
                xt_logits, _, finished = pipeline.model_predict(xt_tokens, xt_logits, timestep, kv_cache)
                xt_tokens = pipeline.sample_logits_to_tokens(xt_logits)[None]
                self.broadcast_tp(xt_logits)
                self.broadcast_tp(finished)
                self.broadcast_tp(xt_tokens)
                if finished[-1]:
                    break

            canvas = pipeline.argmax_logits_to_tokens(xt_logits)
            generated.append(canvas)
            if torch.isin(canvas, pipeline.eos_token_id).any():
                break
            kv_cache = pipeline.build_kv_cache(canvas[None], kv_cache)

        gen_tokens = pipeline.strip_thinking_tokens(torch.cat(generated))
        return pipeline.processor.decode(gen_tokens, skip_special_tokens=True)

    def select_parent_idx(self) -> int:
        # Exploitation: random elite from the top-archive_size by reward; else uniform-random.
        if random.random() < self.config.exploitation_ratio:
            order = sorted(range(len(self.database)), key=lambda i: self.database[i].reward, reverse=True)
            return random.choice(order[: self.config.archive_size])
        return random.randrange(len(self.database))

    def inspirations(self, parent_idx):
        # Deterministic given the archive: top-num_inspirations by reward, excluding the parent.
        order = sorted(range(len(self.database)), key=lambda i: self.database[i].reward, reverse=True)
        picks = [i for i in order if i != parent_idx][: self.config.num_inspirations]
        return [(self.database[i].code, self.database[i].reward) for i in picks]

    def run(self):
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.pipeline.model.eval()

        codes, rewards, generations, artifacts_list = [], [], [], []

        for _ in range(self.N_local):
            # Sync the parent choice across the TP group (tp_main decides); DP rows diverge via their
            # own RNG, giving search diversity. The archive is identical across ranks, so idx aligns.
            idx_t = torch.tensor([self.select_parent_idx()], device=self.accelerator.device)
            self.broadcast_tp(idx_t)
            idx = int(idx_t.item())

            parent = self.database[idx]
            prompt = self.task.build_prompt(parent.code, parent.reward, parent.artifacts, self.inspirations(idx))
            response = self.generate(prompt)
            code = self.task.extract_program(response)
            reward, _, artifacts = self.task.evaluate_program(code)
            reward, artifacts = self.broadcast_object_tp((reward, artifacts))

            codes.append(code or response)
            rewards.append(reward)
            generations.append(parent.generation + 1)
            artifacts_list.append(artifacts)

        rewards = torch.tensor(rewards, device=self.accelerator.device, dtype=torch.float32)

        gathered_codes       = sum(self.gather_object_dp(codes), [])
        gathered_artifacts   = sum(self.gather_object_dp(artifacts_list), [])
        gathered_generations = sum(self.gather_object_dp(generations), [])
        gathered_rewards     = self.gather_dp(rewards)

        for code, r, artifacts, gen in zip(gathered_codes, gathered_rewards.tolist(), gathered_artifacts, gathered_generations):
            program = Program(code, r, {"combined_score": r}, artifacts, generation=gen)
            self.database.append(program)
            if r > self.best.reward:
                self.best = program

        objective_evaluations = epoch * self.config.sample.total_samples
        info = {
            "best-so-far": torch.tensor(self.best.reward, device=self.accelerator.device),
            "valid-fraction": (gathered_rewards > 0).float().mean(),
            "mean-generation": torch.tensor(gathered_generations, dtype=torch.float32).mean(),
        }
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_codes, stage="sampling")
        self.log_info(objective_evaluations=objective_evaluations, info=info, stage="sampling")

    def log_info(self, objective_evaluations, info, stage):
        log_dict = {"objective-evaluations": objective_evaluations}
        for key, value in info.items():
            log_dict[f"info/{stage}/{key}"] = value.item()
        self.accelerator.log(log_dict)


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
