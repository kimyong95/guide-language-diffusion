import math
import sys

import einops
import torch
from absl import flags
from accelerate.utils import broadcast, gather_object
from ml_collections import config_flags
from tqdm import tqdm
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/optimize-noise.py", "Training configuration.")


def sample_noise(mu, sigma, batch_size, device):
    """Draw a batch of diagonal-Gaussian noise for the per-layer KV-noise hidden states, plus its
    sphere-projected version: same direction as the raw sample, but rescaled to the magnitude of the
    underlying standard Gaussian. The projection normalizes each (layer, token) vector independently
    over the hidden D axis (per sample, per layer, per token).

    Args:
        mu, sigma: (H, L, D) mean / variance over the H layers x L noise tokens (each D-dim).
    Returns:
        noise: (B, H, L, D) raw samples mu + sqrt(sigma) * eps -- fed to the optimizer update.
        projected_noise: (B, H, L, D) sphere-projected samples -- fed to the KV injection.
    """
    base_noise = torch.randn(batch_size, *mu.shape, device=device)
    noise = mu[None] + sigma[None] ** 0.5 * base_noise
    projected_noise = noise / torch.linalg.vector_norm(noise, dim=-1, keepdim=True) * torch.linalg.vector_norm(base_noise, dim=-1, keepdim=True)
    return noise, projected_noise


def update_parameters(mu, sigma, noise, objective_values, lr=1.0):
    """Diagonal-Gaussian parameter update (minimize score). Mirrors optimize-epsilon-ref.py's per-slot
    loop, extended to our two leading axes: the H layers x L noise tokens are looped over (each
    (h, l) is its own independent D-dim Gaussian), and the hidden D is the single content axis, kept
    vectorized. Parameters stay in their original (H, L, D) shape -- no flattening.

    Args:
        mu, sigma: (H, L, D) parameters to update.
        noise: (H, L, N, D) samples that produced the objectives.
        objective_values: (H, L, N) score to minimize (i.e. -reward); one reward per sample, broadcast
            across the H x L slots.
    Returns:
        mu, sigma: (H, L, D) updated parameters.
    """
    assert noise.shape[0] == objective_values.shape[0] == mu.shape[0] == sigma.shape[0]
    assert noise.shape[1] == objective_values.shape[1] == mu.shape[1] == sigma.shape[1]
    assert noise.shape[2] == objective_values.shape[2]
    assert noise.shape[3] == mu.shape[2] == sigma.shape[2]

    H, L, N, D = noise.shape  # (H, L) are looped; D is the single hidden axis (vectorized)

    mu = mu.clone()
    sigma = sigma.clone()

    lr_mu = lr
    lr_sigma = lr / math.sqrt(D)

    for h in range(H):
        for l in range(L):

            objective_values_hl = objective_values[h, l]  # (N,); with a shared reward this is just the reward
            noise_hl = noise[h, l]                         # (N, D)
            objective_values_hl_normalized = (objective_values_hl - objective_values_hl.mean()) / objective_values_hl.std().clamp(min=1e-8)

            objective_values_hl_softmaxed = torch.softmax(-objective_values_hl_normalized, dim=0)  # (N,)

            sigma[h, l] = 1 / (

                1/sigma[h, l] + lr_sigma * (

                    (1/sigma[h, l])[None,:] * (noise_hl - mu[h, l, None, :]) * (noise_hl - mu[h, l, None, :]) * (1/sigma[h, l])[None,:] * \

                    objective_values_hl_softmaxed[:,None]

                # sum over N
                ).sum(0)
            )

            mu[h, l] = mu[h, l] - lr_mu * (

                (noise_hl - mu[h, l][None,:]) * \

                objective_values_hl_normalized[:,None]

            # mean over N
            ).mean(0)

    return mu, sigma


class Trainer(BaseTrainer):
    """Noise-search baseline: optimize per-layer random KV rows -- L hidden-state vectors injected into
    every transformer layer's KV cache (cf. test-qwen-noise.py Stage 2) -- to maximize the task reward.
    Each epoch draws N samples from a diagonal Gaussian (mu, sigma), rolls each out deterministically
    (greedy) on top of prompt + noise KV, and updates (mu, sigma) from the rewards. The prompt is
    fixed: the injected noise is the only optimization variable. The epoch / N-per-epoch loop mirrors
    best-of-n."""

    def __init__(self, config):
        super().__init__(config)

        N = self.config.sample.total_samples
        G = self.accelerator.num_processes
        assert N % G == 0, "total_samples must be divisible by num_processes"
        self.N_local = N // G

        # Diagonal-Gaussian noise parameters over the per-layer KV-noise hidden states: mu=0, sigma=1
        # -> the first epoch draws standard-Gaussian directions, exactly like test-qwen-noise.py Stage
        # 2. Shape (H, L, D): each of the H layers gets its own L noise tokens (each D-dim).
        param_shape = (self.model.config.num_hidden_layers, self.config.sample.noise_length, self.model.config.hidden_size)
        self.mu = torch.zeros(*param_shape, device=self.accelerator.device)
        self.sigma = torch.ones(*param_shape, device=self.accelerator.device)

        # Fixed prompt (encoded once, reused every epoch to build the clean KV cache): system + user.
        # It asks for a rewrite of the task's seed code and is never re-built, so the best code the
        # task ratchets along the way never feeds back into it -- the noise is the only variable.
        system_prompt, user_prompt = self.task.prompt()
        self.prompt_tokens = self.build_prompt_tokens(user_prompt, system_prompt=system_prompt, enable_thinking=self.config.sample.enable_thinking)  # (1, P)

    def run(self):
        self.model.eval()
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.accelerator.end_training()

    def build_noisy_cache(self, x_noise):  # x_noise: (H, L, D) sphere-projected noise for one sample
        # Build the clean prompt KV cache, then inject L "random" KV rows into every layer -- each row
        # produced by pushing this sample's Gaussian hidden state through that layer's real K/V
        # pipeline (input_layernorm -> k/v_proj -> k_norm -> RoPE), landing the keys/values on the
        # model's true K/V manifold (cf. test-qwen-noise.py Stage 2). Returns (cache, prompt_logits, P).
        L = self.config.sample.noise_length
        out = self.model(input_ids=self.prompt_tokens, use_cache=True, logits_to_keep=1)  # only the last row is read
        cache = out.past_key_values
        P = cache.get_seq_length()
        pos = torch.arange(P, P + L, device=self.model.device).unsqueeze(0)  # positions P .. P+L-1
        rotary = self.model.model.rotary_emb
        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            dev = attn.k_proj.weight.device
            Dh = attn.head_dim
            x = x_noise[i][None].to(dev, attn.k_proj.weight.dtype)  # (1, L, D)
            xln = layer.input_layernorm(x)  # RMSNorm washes x's scale (magnitude is irrelevant here)
            k = attn.k_norm(einops.rearrange(attn.k_proj(xln), "b l (h d) -> b l h d", d=Dh))  # k_norm over head_dim
            k = einops.rearrange(k, "b l h d -> b h l d")               # (1, Hkv, L, Dh)
            v = einops.rearrange(attn.v_proj(xln), "b l (h d) -> b h l d", d=Dh)  # (1, Hkv, L, Dh)
            cos, sin = rotary(v, pos.to(dev))
            k, _ = apply_rotary_pos_emb(k, k, cos.to(dev), sin.to(dev))  # RoPE at positions P .. P+L-1
            cl = cache.layers[i]
            cl.keys = torch.cat([cl.keys, k.to(cl.keys.device)], dim=2)
            cl.values = torch.cat([cl.values, v.to(cl.values.device)], dim=2)
        return cache, out.logits, P

    def generate(self, x_noise):
        # Greedy AR decode over prompt + this sample's noise KV -- deterministic given the noise, so the
        # noise is the only source of rollout variation. The first token is seeded from the prompt-only
        # logits (it does not attend to the noise: causality forbids position P-1 from seeing rows at
        # >=P); every later token is generated at position >= P+L and does attend over the random KV.
        # Padding the input to P+L+1 makes `generate` feed only first_token (at position P+L) and take
        # over -- the L placeholders stand in for the injected KV rows and are never embedded.
        L = self.config.sample.noise_length
        cache, prompt_logits, P = self.build_noisy_cache(x_noise)
        first_token = prompt_logits[:, -1:, :].argmax(-1)  # (1, 1)
        placeholders = self.prompt_tokens.new_full((1, L), self.tokenizer.pad_token_id)
        input_ids = torch.cat([self.prompt_tokens, placeholders, first_token], dim=1)  # (1, P + L + 1)
        attention_mask = torch.ones_like(input_ids)  # all-ones: the placeholders must not read as padding
        out = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            max_new_tokens=self.config.sample.max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(out[0, P + L:], skip_special_tokens=True)

    @torch.no_grad()
    def sampling_step(self, epoch):
        noise, projected_noise = sample_noise(self.mu, self.sigma, self.N_local, self.accelerator.device)  # (N_local, H, L, D)

        responses = []
        rewards = []
        for b in range(self.N_local):
            response = self.generate(projected_noise[b])  # (H, L, D)
            rewards.append(self.task.evaluate(response))
            responses.append(response)

        rewards = torch.tensor(rewards, device=self.accelerator.device, dtype=torch.float32)

        gathered_noise     = self.accelerator.gather(noise)    # (N, H, L, D) raw noise, gathered along the sample axis
        gathered_responses = gather_object(responses)          # variable-length strings -> object gather
        gathered_rewards   = self.accelerator.gather(rewards)

        # Per-slot Gaussian update over the (H, L) leading axes (mu kept in its original (H, L, D) shape).
        H, L, _ = self.mu.shape
        objective_values = (-gathered_rewards)[None, None].expand(H, L, -1)     # (H, L, N), shared across all slots
        noise_hlnd = einops.rearrange(gathered_noise, "N H L D -> H L N D")
        self.mu, self.sigma = update_parameters(self.mu, self.sigma, noise_hlnd, objective_values, lr=self.config.lr)
        self.mu = broadcast(self.mu)        # keep parameters identical across ranks
        self.sigma = broadcast(self.sigma)

        # parameter-drift diagnostics (logging only): average per-(layer,token) L2 norm of mu, total |sigma|.
        mu_l2_norm = self.mu.norm(dim=-1).mean().item()
        sigma_abs_sum = self.sigma.abs().sum().item()

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling", extra={"sampling/mu-l2-norm": mu_l2_norm, "sampling/sigma-abs-sum": sigma_abs_sum})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_responses, stage="sampling")


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()