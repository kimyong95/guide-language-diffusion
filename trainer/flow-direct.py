import sys
from collections import defaultdict
from pathlib import Path

import torch
import wandb
from absl import flags
from ml_collections import config_flags
from tqdm import tqdm

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/flow-direct.py", "Training configuration.")

def gp_grad(X, Y, x):
    N, d = X.shape[0], X.shape[1:]
    n = x.shape[0]

    X = X.reshape(N, -1).to(dtype=torch.float64)
    Y = Y.reshape(N, 1).to(dtype=torch.float64)
    x = x.reshape(n, -1).to(dtype=torch.float64)

    K      = torch.exp(-torch.cdist(X, X).pow(2) / 2)             # (N, N)
    k_star = torch.exp(-torch.cdist(x, X).pow(2) / 2)             # (n, N)

    L      = torch.linalg.cholesky(K + 1e-6 * torch.eye(N, dtype=torch.float64, device=K.device))
    alpha  = torch.cholesky_solve(Y, L)                         # (N, 1)

    w      = k_star * alpha.T                                   # (n, N)
    y_hat  = k_star @ alpha                                     # (n, 1)
    grad   = w @ X - y_hat * x                                  # (n, D)

    return grad.reshape(n, *d).to(dtype=torch.float32)

class Trainer(BaseTrainer):
    def __init__(self, config):

        super().__init__(config)

        self.question = self.task.question_prompt()

        N = self.config.sample.total_samples
        G = self.accelerator.num_processes
        self.N_local = N // G

        # x1 stores the last-layer hidden state (N, L, H) bf16
        self.data = {
            "x1": torch.empty(0, *self.latent_shape, device=self.accelerator.device, dtype=torch.bfloat16),
            "rewards": torch.empty(0, device=self.accelerator.device),
        }

    def run(self):
        for epoch in tqdm(range(1, self.config.max_epochs+1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.log_data(self.data)
        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.pipeline.transformer.eval()

        info = defaultdict(list)

        timesteps = self.pipeline.scheduler.set_timesteps(
            num_inference_steps=self.config.sample.num_inference_steps,
            device=self.accelerator.device,
        )
        xt = self.pipeline.init_tokens(self.N_local)  # (B, L) long

        for time_i, timestep in enumerate(timesteps):
            pred_latents = self.pipeline.predict_latents(xt, self.question)  # (B, L, H)

            g = self.gp_grad(pred_latents)
            pred_latents = pred_latents + 0.01 * g

            pred_logits = self.latents_to_logits(pred_latents)
            xt = self.pipeline.scheduler.step(xt, pred_logits, timestep)

        x1_texts = self.decode_texts(xt)
        x1_latents = self.tokens_to_latents(xt)
        rewards = self.task.evaluate(x1_texts).to(self.accelerator.device)

        # Final latent via LM-head row lookup (cheap; no extra forward pass).
        gathered_x1_texts   = self.accelerator.gather_for_metrics(x1_texts)
        gathered_x1_latents = self.accelerator.gather(x1_latents)
        gathered_rewards    = self.accelerator.gather(rewards)

        self.increment_data(gathered_x1_latents, gathered_rewards)

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_x1_texts, stage="sampling")
        self.log_info(objective_evaluations=objective_evaluations, info=info, stage="sampling")


    @torch.no_grad()
    def gp_grad(self, xt):
        # Input:
        #   xt: (n, D...)
        # Output:
        #   d: (n, D...)

        X1 = self.data["x1"]
        Y = self.data["rewards"]

        if len(X1) < 2:
            return torch.zeros_like(xt)

        Y = (Y - Y.mean(dim=0, keepdim=True)) / Y.std(dim=0,keepdim=True).clamp(min=1e-3)

        all_xt = self.accelerator.gather(xt)
        mean_norm_X  = X1.reshape(X1.shape[0], -1).float().norm(dim=1).mean().clamp(min=1e-8)
        mean_norm_xt = all_xt.reshape(all_xt.shape[0], -1).float().norm(dim=1).mean().clamp(min=1e-8)

        grad = gp_grad(X1 / mean_norm_X, Y, xt / mean_norm_xt)
        grad = grad * mean_norm_xt

        return grad.to(device=xt.device, dtype=xt.dtype)

    def increment_data(self, x1_latents, rewards):
        self.data["x1"] = torch.cat([self.data["x1"], x1_latents], dim=0)
        self.data["rewards"] = torch.cat([self.data["rewards"], rewards], dim=0)

        torch.cuda.empty_cache()

    def log_data(self, data_dict):
        if not self.accelerator.is_main_process:
            return

        wandb_tracker = self.accelerator.get_tracker("wandb")
        file_path = f"{wandb_tracker.run.dir}/data.pt"
        torch.save(data_dict, file_path)
        artifact = wandb.Artifact(name=Path(__file__).stem, type="data")
        artifact.add_file(file_path)
        wandb_tracker.run.log_artifact(artifact, aliases=[wandb_tracker.run.id])

    def log_info(self, objective_evaluations, info, stage):
        log_dict = {"objective-evaluations": objective_evaluations}
        for key, values in info.items():
            log_dict[f"info/{stage}/{key}"] = torch.stack(values).mean().item()
        self.accelerator.log(log_dict)


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
