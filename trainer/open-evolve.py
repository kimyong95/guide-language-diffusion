import random
import sys
from dataclasses import dataclass, replace

import torch
from absl import flags
from ml_collections import config_flags
from tqdm import tqdm

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/open-evolve.py", "Training configuration.")

FEATURE_BINS = 10               # MAP-Elites grid resolution per axis (complexity, diversity)
DIVERSITY_REFERENCE_SIZE = 20   # reference codes used to score the diversity feature


@dataclass
class Data:
    code: str
    reward: float
    generation: int
    island: int
    migrant: bool = False


class Trainer(BaseTrainer):
    """OpenEvolve-style search (see openevolve-algorithm.md): a MAP-Elites quality-diversity archive
    split into islands. Each iteration cycles to the next island, samples a parent, asks the model to
    full-rewrite it, evaluates the child, and inserts it into that island's feature grid; islands
    exchange elites via ring migration. Replaces flow-guide's gradient guidance; sampling/logging
    mirror flow-guide. Runs as a single process (sequential); the model may be sharded across GPUs."""

    def __init__(self, config):
        super().__init__(config)
        assert self.accelerator.num_processes == 1, "open-evolve runs single-process (sequential generation)"

        self.timesteps = self.pipeline.scheduler.set_timesteps(
            num_inference_steps=self.config.sample.num_inference_steps, device=self.accelerator.device
        )

        # Each island is its own MAP-Elites grid: {feature-cell -> elite Data}.
        self.islands = [dict() for _ in range(self.config.num_islands)]
        self.feature_range = {}      # feature -> [min, max] for min-max binning
        self.reference_codes = []    # first codes seen, reference set for the diversity feature
        self.best = None

        # Seed island 0 with the initial code (deterministic, so identical across ranks).
        init_code = self.task.initial_code()
        self.add(Data(init_code, self.task.evaluate_code(init_code), generation=0, island=0), island=0)

    # --- MAP-Elites archive ---

    def all_data(self):
        return [data for grid in self.islands for data in grid.values()]

    def diversity(self, code):
        # Mean cheap code-distance to the reference set (0 until the set is seeded).
        refs = self.reference_codes
        if not refs:
            return 0.0
        return sum(abs(len(code) - len(r)) * 0.1 + len(set(code) ^ set(r)) * 0.5 for r in refs) / len(refs)

    def cell(self, data):
        # MAP-Elites feature cell: (complexity, diversity), each min-max scaled to a bin index.
        coords = []
        for feature, value in (("complexity", len(data.code)), ("diversity", self.diversity(data.code))):
            rng = self.feature_range.setdefault(feature, [value, value])
            rng[0], rng[1] = min(rng[0], value), max(rng[1], value)
            scaled = 0.0 if rng[1] == rng[0] else (value - rng[0]) / (rng[1] - rng[0])
            coords.append(min(FEATURE_BINS - 1, int(scaled * FEATURE_BINS)))
        return tuple(coords)

    def add(self, child, island):
        # Keep the child only if its cell is empty or it beats the incumbent.
        cell = self.cell(child)
        grid = self.islands[island]
        if cell not in grid or child.reward > grid[cell].reward:
            grid[cell] = child
        child.island = island
        if self.best is None or child.reward > self.best.reward:
            self.best = child
        if len(self.reference_codes) < DIVERSITY_REFERENCE_SIZE:
            self.reference_codes.append(child.code)

    # --- parent / inspiration selection ---

    def sample_parent(self, island):
        data = list(self.islands[island].values())
        if not data:
            return self.best  # empty island -> bootstrap from the global best
        r = random.random()
        if r < self.config.exploration_ratio:
            return random.choice(data)  # explore: uniform within the island
        if r < self.config.exploration_ratio + self.config.exploitation_ratio:
            elites = sorted(self.all_data(), key=lambda p: p.reward, reverse=True)[: self.config.archive_size]
            return random.choice([p for p in elites if p.island == island] or elites)  # exploit: an elite
        return random.choices(data, weights=[max(p.reward, 1e-3) for p in data], k=1)[0]  # weighted

    def sample_inspirations(self, parent, island):
        others = [p for p in self.islands[island].values() if p is not parent]
        if not others:
            return []
        best = max(others, key=lambda p: p.reward)  # island best (quality)
        pool = [p for p in others if p is not best]
        extra = random.sample(pool, min(len(pool), self.config.num_inspirations - 1))  # diversity
        return [(p.code, p.reward) for p in [best, *extra]]

    # --- migration (ring topology) ---

    def migrate(self):
        n = self.config.num_islands
        for i, grid in enumerate(self.islands):
            elites = sorted(grid.values(), key=lambda p: p.reward, reverse=True)
            for migrant in elites[: max(1, int(len(elites) * self.config.migration_rate))]:
                if migrant.migrant:
                    continue  # migrate each data at most once (avoids duplication blow-up)
                for target in ((i + 1) % n, (i - 1) % n):
                    if not any(p.code == migrant.code for p in self.islands[target].values()):
                        self.add(replace(migrant, island=target, migrant=True), island=target)

    # --- generation (full-rewrite; block diffusion, mirrors flow-guide/eval-gemma) ---

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
                xt_logits, finished = pipeline.model_predict(xt_tokens, xt_logits, timestep, kv_cache)
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
        for iteration in tqdm(range(1, self.config.total_iterations + 1), desc="Iterations", position=0, disable=not self.accelerator.is_main_process):
            self.evolve_step(iteration)

        self.accelerator.end_training()

    @torch.no_grad()
    def evolve_step(self, iteration):
        island = (iteration - 1) % self.config.num_islands
        parent = self.sample_parent(island)
        data = [(parent.code, parent.reward)] + self.sample_inspirations(parent, island)
        prompt = self.task.build_prompt(data)

        response = self.generate(prompt)
        code = self.task.extract_code(response)
        reward = self.task.evaluate_code(code)

        child = Data(code, reward, generation=parent.generation + 1, island=island)
        self.add(child, island)
        if iteration % self.config.migration_interval == 0:
            self.migrate()

        rewards = torch.tensor([reward], device=self.accelerator.device, dtype=torch.float32)
        self.log_rewards(objective_evaluations=iteration, rewards=rewards, stage="sampling", extra={"sampling/best-so-far": self.best.reward})
        self.log_texts(objective_evaluations=iteration, rewards=rewards, texts=[code], stage="sampling")


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
