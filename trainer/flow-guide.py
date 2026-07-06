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

def nw_grad(X, Y, x):
    # Batched Nadaraya-Watson gradient  grad_x y_hat(x),  y_hat(x) = sum_n p_n y_n.
    # X: (Q, N, D) per-query support tokens (rms-normed), Y: (N,) rewards, x: (Q, D) queries.
    # All vectors are rms-normed, so ||x - x_n||^2 = 2D - 2 x.x_n and the Gaussian kernel
    # reduces (up to a cancelling constant) to a softmax over scores s_n = x.x_n / sqrt(D).
    # grad = sum_n p_n y_n x_n - y_hat * x_bar = Cov_p(y, x),  x_bar = sum_n p_n x_n.
    Q, N, D = X.shape

    X = X.to(dtype=torch.float32)
    x = x.to(dtype=torch.float32)
    Y = Y.to(dtype=torch.float32)

    s     = torch.einsum("qd,qnd->qn", x, X) / (D**0.5)              # (Q, N) scores
    p     = torch.softmax(s, dim=-1)                                 # (Q, N) kernel weights
    y_hat = p @ Y                                                    # (Q,)
    x_bar = torch.einsum("qn,qnd->qd", p, X)                        # (Q, D) weighted-mean support
    grad  = torch.einsum("qn,qnd->qd", p * Y, X) - y_hat[:, None] * x_bar   # (Q, D)

    return grad / (D**0.5)

class Trainer(BaseTrainer):
    def __init__(self, config):

        super().__init__(config)
        self.guide_enabled = False
        self.init_extention()

        N = self.config.sample.total_samples
        G = self.dp_size
        assert N % G == 0, "total_samples must be divisible by dp_size"
        self.N_local = N // G

        self.prompt = self.task.prompt()

        self.num_layers = len(self.pipeline.model.model.decoder.layers)  # H
        self.data = {
            "x1_hs": torch.empty(0, len(self.config.guidance_layers), self.pipeline.gen_length, self.pipeline.hidden_size, device=self.accelerator.device, dtype=torch.bfloat16),  # (N, G, L, D)
            "rewards": torch.empty(0, device=self.accelerator.device),
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
            g = external_self.nw_grad(x_normed, self.layer_idx).to(device=x.device, dtype=x.dtype)  # (n, L, D)
            external_self.info[f"g-norm-{self.layer_idx}"].append(g.float().norm(dim=(-2, -1)))  # (n,)
            x_guided = x + g * external_self.config.guide_scale
            x = x_guided * (x.norm(dim=-1, keepdim=True) / x_guided.norm(dim=-1, keepdim=True).clamp(min=1e-6))

        return original_forward(x, *args, **kwargs)

    def run(self):
        for epoch in tqdm(range(1, self.config.max_epochs+1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.pipeline.model.eval()

        self.info = defaultdict(list)

        timesteps = self.pipeline.scheduler.set_timesteps(num_inference_steps=self.config.sample.num_inference_steps,device=self.accelerator.device)

        # same prompt for every sample; encode once, reused read-only across the loop below
        prompt_tokens = self.pipeline.build_prompt_tokens(self.prompt)
        kv_cache = self.pipeline.build_kv_cache(prompt_tokens)

        x1_texts = []  # one (variable-length) completion text per sample
        x1_hidden_states = torch.empty(self.N_local, len(self.config.guidance_layers), self.pipeline.gen_length, self.pipeline.hidden_size, device=self.accelerator.device, dtype=torch.bfloat16,)  # (N_local, G, L, D)

        for i in range(self.N_local):  # one sequence at a time
            xt_logits = None
            xt_tokens = self.pipeline.sample_init_tokens()[None]
            with self.enable_guide():
                for timestep in timesteps:
                    xt_logits, hidden_states, finished = self.pipeline.model_predict(xt_tokens, xt_logits, timestep, kv_cache)  # (L, V)
                    xt_tokens = self.pipeline.sample_logits_to_tokens(xt_logits)[None]
                    if finished[-1]:
                        break
            x1_texts.append(self.pipeline.argmax_logits_to_text(xt_logits))  # one completion string per sample

            # Reference hidden states: one clean, unguided pass over the final answer tokens.
            x1_tokens = self.pipeline.argmax_logits_to_tokens(xt_logits)  # (L,)
            _, hidden_states, _ = self.pipeline.model_predict(x1_tokens[None], None, timesteps[-1], kv_cache)  # (1, L)
            x1_hidden_states[i] = rms_norm(hidden_states)[list(self.config.guidance_layers)]  # (G, L, D)

        rewards = self.task.evaluate(x1_texts).to(self.accelerator.device)

        gathered_x1_hidden_states = self.gather(x1_hidden_states)
        gathered_x1_texts         = self.gather_object(x1_texts)
        gathered_rewards          = self.gather(rewards)
        gathered_info             = {key: self.gather(torch.cat(values).mean().reshape(1)) for key, values in self.info.items()}

        self.increment_data(gathered_x1_hidden_states, gathered_rewards)

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_x1_texts, stage="sampling")
        self.log_info(objective_evaluations=objective_evaluations, info=gathered_info, stage="sampling")


    @torch.no_grad()
    def nw_grad(self, xt, layer_id):
        """
        Args:
            xt: (n, L, D) input to layer layer_id, already rms_norm'd by the caller.
            layer_id: actual layer index (must be in config.guidance_layers); resolved to a storage position internally.
        Returns:
            grad: (n, L, D) per-position guidance direction.
        """
        n, L, D = xt.shape

        l = self.config.guidance_layers.index(layer_id)
        X1 = self.data["x1_hs"][:, l].float()   # (N, L, D) support tokens, already rms_norm'd
        Y = self.data["rewards"]                # (N,)
        N = len(X1)

        if N < 2:
            return torch.zeros(n, L, D, device=xt.device, dtype=xt.dtype)

        Yz = (Y - Y.mean()) / Y.std().clamp(min=1e-3)                 # std-normalize reward scale
        q = xt.reshape(n * L, D).float().to(self.accelerator.device)  # (Q, D) query tokens

        idx  = torch.einsum("qd,nld->qnl", q, X1).argmax(dim=-1)      # (Q, N) closest token per sentence
        X_nn = X1[torch.arange(N, device=X1.device), idx]            # (Q, N, D) selected support
        grad = nw_grad(X_nn, Yz, q)                                  # (Q, D) batched standalone

        return grad.reshape(n, L, D).to(device=xt.device, dtype=xt.dtype)

    def increment_data(self, x1_hidden_states, rewards):
        self.data["x1_hs"] = torch.cat([self.data["x1_hs"], x1_hidden_states.to(self.accelerator.device)], dim=0)
        self.data["rewards"] = torch.cat([self.data["rewards"], rewards.to(self.accelerator.device)], dim=0)

        torch.cuda.empty_cache()


    def log_info(self, objective_evaluations, info, stage):
        log_dict = {"objective-evaluations": objective_evaluations}
        for key, values in info.items():
            log_dict[f"info/{stage}/{key}"] = values.mean().item()
        self.accelerator.log(log_dict)


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
