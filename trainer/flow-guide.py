import sys
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import einops
import torch
import wandb
from absl import flags
from ml_collections import config_flags
from tqdm import tqdm

from base import BaseTrainer
from utils import batch_slices

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/flow-guide.py", "Training configuration.")

def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS-normalization
    Args:
        x: (B, L, d) hidden states
    Returns:
        x: (B, L, d) normalized hidden states
    """
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def gp_grad(X, Y, x):
    # X: (N, d) data points, Y: (N,) rewards, x: (n, d) query points
    # returns grad: (n, d)
    N, d = X.shape[0], X.shape[1]

    X = X.to(dtype=torch.float64)
    Y = einops.rearrange(Y, "N -> N 1").to(dtype=torch.float64)
    x = x.to(dtype=torch.float64)

    K      = torch.exp(-torch.cdist(X, X) ** 2 / 2 / (d**0.5))  # (N, N)
    k_star = torch.exp(-torch.cdist(x, X) ** 2 / 2 / (d**0.5))  # (n, N)

    L      = torch.linalg.cholesky(K + 1e-6 * torch.eye(N, dtype=torch.float64, device=K.device))
    alpha  = torch.cholesky_solve(Y, L)                         # (N, 1)

    w      = k_star * alpha.T                                   # (n, N)
    y_hat  = k_star @ alpha                                     # (n, 1)
    grad   = w @ X - y_hat * x                                  # (n, d)

    return grad.to(dtype=torch.float32)

class Trainer(BaseTrainer):
    def __init__(self, config):

        super().__init__(config)
        self.enable_guide = True
        self.init_extention()

        N = self.config.sample.total_samples
        G = self.accelerator.num_processes
        self.N_local = N // G

        self.question = self.task.question_prompt()
        self.question_tokens = self.pipeline.encode_prompt(self.question)[0]

        self.data = {
            "x1_hs": torch.empty(0, self.pipeline.hidden_size, device=self.accelerator.device, dtype=torch.bfloat16),
            "rewards": torch.empty(0, device=self.accelerator.device),
        }

    def init_extention(self):
        guidance_block = self.pipeline.transformer.model.transformer.blocks[self.config.guide_layer_id]

        guidance_block.forward_original = guidance_block.forward
        guidance_block.forward = partial(Trainer.extended_llada_block_forward, guidance_block, external_self=self)

    @contextmanager
    def guide_disabled(self):
        prev = self.enable_guide
        self.enable_guide = False
        try:
            yield
        finally:
            self.enable_guide = prev
    
    # Extension to LLaDALlamaBlock.forward
    @staticmethod
    @torch.no_grad()
    def extended_llada_block_forward(
        self,                      # the LLaDALlamaBlock instance
        x,                         # (B, L, d_model) block input hidden state
        attention_bias=None,
        layer_past=None,
        use_cache=False,
        external_self=None,        # -> the Trainer (this `self`)
    ):
        if external_self.enable_guide:
            P = len(external_self.question_tokens)
            g = external_self.gp_grad(x[:, P:, :]).to(device=x.device, dtype=x.dtype)
            x[:, P:, :] = x[:, P:, :] + g
        return self.forward_original(x,attention_bias=attention_bias,layer_past=layer_past,use_cache=use_cache)

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

        x1_tokens = self.pipeline.init_tokens(self.N_local)  # (N_local, L) long
        x1_hidden_states = torch.empty(self.N_local, self.pipeline.hidden_size,device=self.accelerator.device, dtype=torch.bfloat16,)

        for sl in batch_slices(self.N_local, self.config.sample.max_batch_size_per_device):
            xt_tokens = self.pipeline.init_tokens(sl.stop - sl.start)  # (b, L) long
            for time_i, timestep in enumerate(timesteps):
                logits, _ = self.pipeline.model_predict(xt_tokens, self.question_tokens)  # (b, L, V)
                xt_tokens = self.pipeline.scheduler.step(xt_tokens, logits, timestep)

            # generate reference data
            with self.guide_disabled():
                _, hidden_states = self.pipeline.model_predict(xt_tokens, self.question_tokens)  # (H, b, L, d)
            x1_tokens[sl] = xt_tokens
            hidden_states = hidden_states[self.config.guide_layer_id] # (B, L, d)
            hidden_states = rms_norm(hidden_states)
            x1_hidden_states[sl] = einops.reduce(hidden_states, "B L D -> B D", "mean")

        x1_texts = self.tokens_to_text(x1_tokens)
        rewards = self.task.evaluate(x1_texts).to(self.accelerator.device)

        gathered_x1_hidden_states = self.accelerator.gather(x1_hidden_states)
        gathered_x1_texts         = self.accelerator.gather_for_metrics(x1_texts)
        gathered_rewards          = self.accelerator.gather(rewards)

        self.increment_data(gathered_x1_hidden_states, gathered_rewards)

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_x1_texts, stage="sampling")
        self.log_info(objective_evaluations=objective_evaluations, info=info, stage="sampling")


    @torch.no_grad()
    def gp_grad(self, xt):
        # Input:
        #   xt: (n, L, D)  per-token hidden states
        # Output:
        #   grad: (n, L, D)  same update direction for every token l

        n, L, D = xt.shape

        X1 = self.data["x1_hs"]    # (N, D) mean-pooled hidden states
        Y = self.data["rewards"]   # (N,)

        if len(X1) < 2:
            return torch.zeros_like(xt)
        
        xt = rms_norm(xt)
        Y = (Y - Y.mean(dim=0, keepdim=True)) / Y.std(dim=0,keepdim=True).clamp(min=1e-3)

        # the GP lives in the mean-pooled (D,) space, so pool the query over L
        x = einops.reduce(xt, "n L D -> n D", "mean")

        grad = gp_grad(X1, Y, x)   # (n, D)

        # broadcast the same update direction to every token l in L
        grad = einops.repeat(grad, "n D -> n L D", L=L)

        return grad.to(device=xt.device, dtype=xt.dtype)

    def increment_data(self, x1_hidden_states, rewards):
        self.data["x1_hs"] = torch.cat([self.data["x1_hs"], x1_hidden_states], dim=0)
        self.data["rewards"] = torch.cat([self.data["rewards"], rewards], dim=0)

        # self.pipeline.transformer.model.transformer.blocks[0].attn_norm.weight

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
