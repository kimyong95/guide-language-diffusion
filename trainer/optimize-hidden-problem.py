"""CMA-ES over the hidden-state intervention x, on a single problem prompt -- no dataset.

    for each epoch:
        solutions = es.ask()                                    # N candidates, each a flat (Lx*D,) vector
        rollouts = [generate(prompt, x) for x in solutions]
        x_hats = [optimize(x, rollout) for x, rollout in ...]   # gradient fit of x to the tokens it produced
        es.tell(x_hats, -rewards)

es is told the fitted point, not the drawn one: the reward comes from the draw and the coordinates from
its refinement, so the rollout reaches the search as tokens and not only as a scalar. The two stay
coherent because x_hat is fitted to reproduce the very rollout that earned the reward.

The whole intervention is one search vector: the Lx slots are concatenated, so the optimizer never sees
the (Lx, D) structure. A candidate is projected row-wise back onto the sqrt(D) sphere before it reaches
the model -- the manifold the intervention lives on -- and every inner step reprojects, so what es is
told never leaves it: evaluation is invariant to the radius, so telling an off-sphere point would let
the mean random-walk along a direction the reward cannot see.

Sampling is greedy, so x is the only source of randomness: one candidate gives one deterministic rollout
and one reward. Rank 0's population is broadcast every epoch and told back to every es in ask order, so
the search never rests on the ranks drawing identical RNG; a rank only evaluates its own contiguous
slice of the population.
"""

import math
import sys

import cma
import einops
import numpy as np
import torch
from absl import flags
from accelerate.utils import broadcast, gather_object
from ml_collections import config_flags
from tqdm import tqdm

import problems
from base import BaseTrainer
from utils import batch_slices, concat

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/optimize-hidden-problem.py", "Training configuration.")


class Trainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        if config.train.gradient_checkpointing:
            # non-reentrant: x is written in by a layer hook, not passed in as a checkpointed input
            self.pipeline.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        assert config.sample.total_samples % self.accelerator.num_processes == 0, f"population ({config.sample.total_samples}) must be divisible by number of GPUs ({self.accelerator.num_processes})"
        self.N_local = config.sample.total_samples // self.accelerator.num_processes

        D = self.pipeline.config.hidden_size
        self.Lx = config.sample.n_intervene
        generator = torch.Generator().manual_seed(config.seed)
        x_init = self.project_to_sphere(torch.randn(self.Lx, D, generator=generator))
        self.es = cma.CMAEvolutionStrategy(x_init.flatten().numpy(), config.train.sigma, {
            "popsize": config.sample.total_samples,
            "seed": config.seed + 1,      # pycma reads 0 as "seed from the clock"
            "CMA_diagonal": True,         # a full covariance over Lx*D coordinates is (Lx*D)^2 floats
            "verbose": -9,
        })

    def setup_task(self):
        self.problem = problems.get_problem(self.config.problem)

    @staticmethod
    def project_to_sphere(x):
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True) * math.sqrt(x.shape[-1])

    @torch.enable_grad()
    def optimize_x(self, prompt_tokens, generated_tokens, x_init):
        """
        Args:
            prompt_tokens: list (Lp)
            generated_tokens: list (Lg), what x_init itself generated
            x_init: (Lx, D)

        Returns:
            (Lx, D) on the sqrt(D) sphere
        """
        cfg = self.config.train
        x = x_init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=cfg.learning_rate)
        for _ in range(cfg.optimize_steps):
            optimizer.zero_grad()
            with self.pipeline.intervene([prompt_tokens + generated_tokens], x[None]):
                loss = -self.pipeline.log_probs(prompt_tokens, generated_tokens).mean()
                loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.copy_(self.project_to_sphere(x))
        return x.detach()

    def run(self):
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        cfg = self.config.sample

        x_global = self.es.ask()                                                                          # list (N) of (Lx*D,)
        x_global = broadcast(torch.from_numpy(np.array(x_global)).to(self.accelerator.device))            # (N, Lx*D) float64, rank 0's population
        x_global = self.project_to_sphere(einops.rearrange(x_global, "n (lx d) -> n lx d", lx=self.Lx))   # (N, Lx, D)
        x_local = self.ungather(x_global).float()                                                         # (N_local, Lx, D)

        sampling_data = []
        for batch in tqdm(list(batch_slices(self.N_local, cfg.max_batch_size_per_device)), desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            x = x_local[batch]
            prompts = [self.problem.prompt()] * len(x)
            prompt_tokens = self.pipeline.texts_to_tokens(prompts, system_prompt=self.problem.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking, n_intervene=self.Lx)  # 2D list (N_local_batch, Lp)

            self.pipeline.model.eval()   # checkpointing is gated on module.training: generate must not wrap all H layers on every decode step
            generated_output = self.pipeline.generate(prompt_tokens, x, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)
            generated_texts = generated_output.texts                                                                                                        # N_local_batch x str
            rewards = torch.tensor([self.problem.evaluate(text) for text in generated_texts], device=self.accelerator.device, dtype=torch.float32)

            self.pipeline.model.train()  # arms gradient checkpointing; the base weights stay frozen and Qwen3 has no dropout
            x_hat = torch.stack([self.optimize_x(prompt, generated, x_init) for prompt, generated, x_init in zip(prompt_tokens, generated_output.tokens, x)])   # (N_local_batch, Lx, D)

            sampling_data.append({
                "generated_texts": generated_texts,   # N_local_batch x str
                "rewards": rewards,                   # (N_local_batch,)
                "x_hat": x_hat,                       # (N_local_batch, Lx, D)
            })

        sampling_data = {key: concat([batch[key] for batch in sampling_data]) for key in sampling_data[0]}

        gathered_rewards = self.accelerator.gather(sampling_data["rewards"])   # (N,) back in ask order: rank r held candidates [r*N_local, (r+1)*N_local)
        gathered_texts = gather_object(sampling_data["generated_texts"])
        gathered_x_hat = self.accelerator.gather(sampling_data["x_hat"]).double()   # (N, Lx, D) same order
        self.es.tell(list(einops.rearrange(gathered_x_hat, "n lx d -> n (lx d)").cpu().numpy()), (-gathered_rewards).tolist())   # CMA-ES minimizes; the point told is the fit to the rollout the draw produced

        objective_evaluations = epoch * cfg.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_texts, stage="sampling")


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
