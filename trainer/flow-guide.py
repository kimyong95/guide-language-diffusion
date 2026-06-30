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

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/flow-guide.py", "Training configuration.")

def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS-normalization
    Args:
        x: (n, L, D) hidden states
    Returns:
        x: (n, L, D) normalized hidden states
    """
    return x * torch.rsqrt((x.float()**2).mean(dim=-1, keepdim=True) + eps).type_as(x)

def gp_alpha(X, Y):
    # X: (N, M) support points, Y: (N,) rewards  (M = L*D feature dim)
    # returns alpha: (N, 1) dual weights = (K + eps I)^-1 Y
    # The O(N^2 M) cdist + Cholesky here depend only on the (fixed-per-epoch) support
    # set, so this is cached and reused across every query of an epoch. Kept in float64.
    N, M = X.shape

    X = X.to(dtype=torch.float64)
    Y = einops.rearrange(Y, "N -> N 1").to(dtype=torch.float64)

    K     = torch.exp(-torch.cdist(X, X) ** 2 / 2 / M)  # (N, N)
    chol  = torch.linalg.cholesky(K + 1e-6 * torch.eye(N, dtype=torch.float64, device=K.device))
    alpha = torch.cholesky_solve(Y, chol)               # (N, 1)

    return alpha.to(dtype=torch.float32)


def gp_grad(X, alpha, x):
    # X: (N, M) support points, alpha: (N, 1) dual weights, x: (n, M) query points
    # returns grad: (n, M)
    # Per-query O(n N M) part. cdist stays accurate enough in float32 thanks to the /2M
    # normalization; matmuls run in float32 since the caller renormalizes the direction.
    N, M = X.shape

    X = X.to(dtype=torch.float32)
    x = x.to(dtype=torch.float32)

    k_star = torch.exp(-torch.cdist(x, X) ** 2 / 2 / M)  # (n, N)

    w      = k_star * alpha.T                             # (n, N)
    y_hat  = k_star @ alpha                               # (n, 1)
    grad   = w @ X - y_hat * x                            # (n, M)

    return grad / (M**0.5) / (N**0.5)

class Trainer(BaseTrainer):
    def __init__(self, config):

        super().__init__(config)
        self.guide_enabled = False
        self.init_extention()

        N = self.config.sample.total_samples
        G = self.accelerator.num_processes
        self.N_local = N // G

        self.prompt = self.task.prompt()

        self.num_layers = len(self.pipeline.model.model.decoder.layers)  # H
        self.data = {
            "x1_hs": torch.empty(0, len(self.config.guidance_layers), self.pipeline.gen_length, self.pipeline.hidden_size, device=self.accelerator.device, dtype=torch.bfloat16),  # (N, G, L, D)
            "rewards": torch.empty(0, device=self.accelerator.device),
            "alpha": {},  # layer_id -> dual weights; cleared whenever the support set grows (see increment_data)
        }

    def init_extention(self):
        for block in self.pipeline.model.model.decoder.layers:
            # Wrap each layer's forward: guide the block input, then call the unmodified layer.
            block.forward = partial(
                Trainer.extended_decoder_layer_forward,
                block,
                external_self=self,
                original_forward=block.forward,
            )

    @contextmanager
    def enable_guide(self):
        prev = self.guide_enabled
        self.guide_enabled = True
        try:
            yield
        finally:
            self.guide_enabled = prev
    
    @staticmethod
    @torch.no_grad()
    def extended_decoder_layer_forward(
        self,
        hidden_states,
        *args,
        external_self=None,
        original_forward=None,
        **kwargs,
    ):
        """Guidance sublayer wrapping each decoder layer (installed by init_extention).

        Args:
            self: the decoder layer instance.
            hidden_states: (n, L, D) block input.
            external_self: the Trainer.
            original_forward: the layer's unmodified forward.
        """
        x = hidden_states
        if external_self.guide_enabled and self.layer_idx in external_self.config.guidance_layers:
            x_normed = rms_norm(x)
            g = external_self.gp_grad(x_normed, self.layer_idx).to(device=x.device, dtype=x.dtype)  # (n, L, D)
            external_self.info[f"g-norm-{self.layer_idx}"].append(g.float().norm(dim=(-2, -1)))  # (n,)
            x_guided = x + g * external_self.config.guide_scale
            x = x_guided * (x.norm(dim=-1, keepdim=True) / x_guided.norm(dim=-1, keepdim=True).clamp(min=1e-6))

        return original_forward(x, *args, **kwargs)

    def run(self):
        for epoch in tqdm(range(1, self.config.max_epochs+1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.log_data(self.data)
        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.pipeline.model.eval()

        self.info = defaultdict(list)

        timesteps = self.pipeline.scheduler.set_timesteps(num_inference_steps=self.config.sample.num_inference_steps,device=self.accelerator.device)

        # same prompt for every sample; encode once, unguided
        self.pipeline.set_prompt(self.prompt)

        x1_texts = []  # one (variable-length) completion text per sample
        x1_hidden_states = torch.empty(self.N_local, len(self.config.guidance_layers), self.pipeline.gen_length, self.pipeline.hidden_size, device=self.accelerator.device, dtype=torch.bfloat16,)  # (N_local, G, L, D)

        for i in range(self.N_local):  # one sequence at a time
            xt_logits = None  # (L, V)
            with self.enable_guide():
                for timestep in timesteps:
                    xt_logits, hidden_states, early_stop = self.pipeline.model_predict_step(xt_logits, timestep)  # (L, V)
                    if early_stop:
                        break
            x1_texts.append(self.pipeline.argmax_logits_to_text(xt_logits))  # one completion string per sample

            # Reference hidden states: one clean, unguided pass over the final answer tokens
            x1_tokens = torch.argmax(xt_logits, dim=-1)  # (L,)
            out = self.pipeline.model(input_ids=None,past_key_values=self.pipeline.kv_cache,decoder_position_ids=self.pipeline.dec_pos,decoder_input_ids=x1_tokens[None],self_conditioning_logits=None,output_hidden_states=True,)  # (1, L)
            hidden_states = torch.stack(out.hidden_states, dim=0)[:, 0]  # (H+1, L, D)
            x1_hidden_states[i] = rms_norm(hidden_states)[list(self.config.guidance_layers)]  # (G, L, D)

        rewards = self.task.evaluate(x1_texts).to(self.accelerator.device)

        gathered_x1_hidden_states = self.accelerator.gather(x1_hidden_states)
        gathered_x1_texts         = self.accelerator.gather_for_metrics(x1_texts)
        gathered_rewards          = self.accelerator.gather(rewards)
        gathered_info             = {key: self.accelerator.gather(torch.cat(values).mean().reshape(1)) for key, values in self.info.items()}

        self.increment_data(gathered_x1_hidden_states, gathered_rewards)

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_x1_texts, stage="sampling")
        self.log_info(objective_evaluations=objective_evaluations, info=gathered_info, stage="sampling")


    @torch.no_grad()
    def gp_grad(self, xt, layer_id):
        """
        Args:
            xt: (n, L, D) input to layer layer_id, already rms_norm'd by the caller.
            layer_id: actual layer index (must be in config.guidance_layers); resolved to a storage position internally.
        Returns:
            grad: (n, L, D) per-position guidance direction.
        """
        n, L, D = xt.shape

        l = self.config.guidance_layers.index(layer_id)
        X1 = self.data["x1_hs"][:, l]   # (N, L, D), already rms_norm'd
        Y = self.data["rewards"]               # (N,)

        if len(X1) < 2:
            return torch.zeros(n, L, D, device=xt.device, dtype=xt.dtype)

        X1 = X1.reshape(len(X1), L * D)                            # (N, L*D)
        x = xt.reshape(n, L * D).to(self.accelerator.device)      # (n, L*D)

        alpha = self.data["alpha"].get(layer_id)
        if alpha is None:  # cache miss: O(N^2 M) solve, reused for every query this epoch
            Yz = (Y - Y.mean(dim=0, keepdim=True)) / Y.std(dim=0, keepdim=True).clamp(min=1e-3)
            alpha = gp_alpha(X1, Yz)
            self.data["alpha"][layer_id] = alpha

        grad = gp_grad(X1, alpha, x)   # (n, L*D)

        return grad.reshape(n, L, D).to(device=xt.device, dtype=xt.dtype)

    def increment_data(self, x1_hidden_states, rewards):
        self.data["x1_hs"] = torch.cat([self.data["x1_hs"], x1_hidden_states.to(self.accelerator.device)], dim=0)
        self.data["rewards"] = torch.cat([self.data["rewards"], rewards.to(self.accelerator.device)], dim=0)
        self.data["alpha"] = {}  # support set changed -> invalidate cached dual weights

        torch.cuda.empty_cache()

    def log_data(self, data_dict):
        if not self.accelerator.is_main_process:
            return

        wandb_tracker = self.accelerator.get_tracker("wandb")
        file_path = f"{wandb_tracker.run.dir}/data.pt"
        # save the support set on CPU (portable); the alpha cache is transient
        save_dict = {"x1_hs": data_dict["x1_hs"].cpu(), "rewards": data_dict["rewards"].cpu()}
        torch.save(save_dict, file_path)
        artifact = wandb.Artifact(name=Path(__file__).stem, type="data")
        artifact.add_file(file_path)
        wandb_tracker.run.log_artifact(artifact, aliases=[wandb_tracker.run.id])

    def log_info(self, objective_evaluations, info, stage):
        log_dict = {"objective-evaluations": objective_evaluations}
        for key, values in info.items():
            log_dict[f"info/{stage}/{key}"] = values.mean().item()
        self.accelerator.log(log_dict)


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
