import math
import sys

import einops
import torch
from absl import flags
from accelerate.utils import broadcast, gather_object
from ml_collections import config_flags
from tqdm import tqdm

from base import BaseTrainer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/optimize-noise.py", "Training configuration.")


def sample_noise(mu, sigma, batch_size, device):
    """Draw a batch of diagonal-Gaussian model noise, one hidden state per full_attention layer.

    Args:
        mu, sigma: (H, D...) per-layer mean / variance.
    Returns:
        noise: (H, B, D...) raw samples mu + sqrt(sigma) * eps.
    """
    D = mu.shape[1:]
    base_noise = torch.randn(mu.shape[0], batch_size, *D, device=device)
    return mu[:, None] + sigma[:, None] ** 0.5 * base_noise


def update_parameters(mu, sigma, noise, objective_values, lr=1.0):
    """Diagonal-Gaussian parameter update (minimize score). Ported from optimize-noise-ref.py with
    the leading axis re-read as full_attention layers instead of diffusion timesteps -- the objective
    is shared across layers here, so the ref's causal `objective_values[t:].mean(0)` is a no-op.

    Args:
        mu, sigma: (H, L, D) parameters to update.
        noise: (H, N, L, D) samples that produced the objectives.
        objective_values: (H, N) score to minimize (i.e. -reward).
    Returns:
        mu, sigma: (H, L, D) updated parameters.
    """
    assert noise.shape[0] == objective_values.shape[0] == mu.shape[0] == sigma.shape[0]
    assert noise.shape[1] == objective_values.shape[1]
    assert noise.shape[2:] == mu.shape[1:] == sigma.shape[1:]

    H, N, L, D = noise.shape
    E = L * D  # internal noise-update dimension: token (L) and hidden (D) axes concatenated

    mu = mu.clone().reshape(H, E)
    sigma = sigma.clone().reshape(H, E)
    noise = noise.reshape(H, N, E)

    lr_mu = lr
    lr_sigma = lr / math.sqrt(E)

    for t in range(H):

        objective_values_t = objective_values[t:].mean(0)  # (N,)
        noise_t = noise[t]                                  # (N, E)
        objective_values_t_normalized = (objective_values_t - objective_values_t.mean()) / objective_values_t.std().clamp(min=1e-8)

        objective_values_t_softmaxed = torch.softmax(-objective_values_t_normalized, dim=0)  # (N,)

        sigma[t] = 1 / (

            1/sigma[t] + lr_sigma * (

                (1/sigma[t])[None,:] * (noise_t - mu[t,None,:]) * (noise_t - mu[t,None,:]) * (1/sigma[t])[None,:] * \

                objective_values_t_softmaxed[:,None]

            # sum over N
            ).sum(0)
        )

        mu[t] = mu[t] - lr_mu * (

            (noise_t - mu[t][None,:]) * \

            objective_values_t_normalized[:,None]

        # mean over N
        ).mean(0)

    return mu.reshape(H, L, D), sigma.reshape(H, L, D)


class Trainer(BaseTrainer):
    """Noise-search baseline: optimize a Gaussian "model noise" -- one hidden state per global
    (full_attention) encoder layer, appended to that layer's KV cache (cf. test-infill) -- to maximize
    the task reward. Each epoch draws N samples from a diagonal Gaussian (mu, sigma), rolls each out
    deterministically, and updates (mu, sigma) from the rewards. The prompt is fixed: the injected
    noise is the only optimization variable. The epoch / N-per-epoch loop mirrors best-of-n."""

    def __init__(self, config):
        super().__init__(config)

        N = self.config.sample.total_samples
        G = self.accelerator.num_processes
        assert N % G == 0, "total_samples must be divisible by num_processes"
        self.N_local = N // G

        self.timesteps = self.pipeline.scheduler.set_timesteps(
            num_inference_steps=self.config.sample.num_inference_steps, device=self.accelerator.device
        )

        # One optimizable model-noise hidden state per global (full_attention) layer; the same noise is
        # injected for every denoising timestep of a generation (cf. test-infill).
        layer_types = self.pipeline.model.config.text_config.layer_types
        self.full_attn_layer_ids = [i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention"]

        # Diagonal-Gaussian noise parameters, indexed by full_attention layer: mu=0, sigma=1 -> the
        # first epoch draws standard-Gaussian model noise, exactly like test-infill.
        D = (self.config.sample.noise_length, self.pipeline.hidden_size)
        self.mu = torch.zeros(len(self.full_attn_layer_ids), *D, device=self.accelerator.device)
        self.sigma = torch.ones(len(self.full_attn_layer_ids), *D, device=self.accelerator.device)

        # Fixed prompt (encoded once, reused every epoch): system + user turns.
        system_prompt, user_prompt = self.task.prompt()
        self.prompt_tokens = self.pipeline.build_prompt_tokens(user_prompt, system_prompt=system_prompt, enable_thinking=self.config.sample.enable_thinking)

        # Best reward seen so far, for logging only (does not affect the noise search). Seeded with the
        # fixed prompt's ref_code (the initial code), evaluated once -- deterministic across ranks.
        init_code = self.task.initial_code()
        self.best = (init_code, self.task.evaluate_code(init_code))  # (code, reward), updated per epoch

    @torch.no_grad()
    def inject_noise(self, kv_cache, noise):
        # Append the model noise to the global (full_attention) layers' KV cache (cf. test-infill):
        # project each layer's Gaussian hidden state through that layer's own K/V pipeline
        # (input_layernorm -> k_proj -> k_norm / v_norm, no RoPE) and concatenate onto the cached
        # keys/values. Sliding layers keep the prompt-only cache, so the decoder canvas still sits
        # right after the prompt. Mutates kv_cache in place.
        enc_layers = self.pipeline.model.model.encoder.language_model.layers
        L_noise = noise.shape[1]
        for h, layer_idx in enumerate(self.full_attn_layer_ids):
            enc_layer = enc_layers[layer_idx]
            cache_layer = kv_cache.layers[layer_idx]
            attn = enc_layer.self_attn
            x = noise[h][None].to(device=attn.k_proj.weight.device, dtype=attn.k_proj.weight.dtype)  # (1, L_noise, d)
            kv = attn.k_proj(enc_layer.input_layernorm(x)).view(1, L_noise, -1, attn.head_dim)   # (1, L_noise, Hkv, d)
            K = attn.k_norm(kv).transpose(1, 2)  # (1, Hkv, L_noise, d), no RoPE
            V = attn.v_norm(kv).transpose(1, 2)  # (1, Hkv, L_noise, d)
            cache_layer.keys = torch.cat([cache_layer.keys, K.to(cache_layer.keys.device)], dim=2)
            cache_layer.values = torch.cat([cache_layer.values, V.to(cache_layer.values.device)], dim=2)

    @torch.no_grad()
    def generate(self, noise):
        # Block diffusion with `noise` injected into the global-layer KV cache. Deterministic given
        # `noise`: all-<pad> init canvas + greedy argmax + a direct model call (no model_predict, no
        # temperature, no stochastic sampling), mirroring test-infill.
        pipeline = self.pipeline
        kv_cache = pipeline.build_kv_cache(self.prompt_tokens)  # fresh cache per sample; grown per block
        self.inject_noise(kv_cache, noise)

        generated = []
        for _ in range(self.config.sample.max_blocks):
            xt_logits = None
            xt_tokens = torch.zeros(1, pipeline.canvas_length, dtype=torch.long, device=self.accelerator.device)  # all <pad> (id 0)
            # decoder canvas sits right after the cached context: positions P .. P+L-1 (P from a sliding layer)
            P = kv_cache.get_seq_length()
            decoder_position_ids = torch.arange(P, P + pipeline.canvas_length, device=self.accelerator.device).unsqueeze(0)
            for _ in self.timesteps:
                out = pipeline.model(
                    input_ids=None,
                    past_key_values=kv_cache,
                    decoder_position_ids=decoder_position_ids,
                    decoder_input_ids=xt_tokens,
                    self_conditioning_logits=xt_logits,
                )
                xt_logits = out.logits[0].to(self.accelerator.device)      # (L, V) raw logits, no temperature
                xt_tokens = pipeline.argmax_logits_to_tokens(xt_logits)[None]  # greedy argmax, (1, L)
            canvas = pipeline.argmax_logits_to_tokens(xt_logits)  # (L,)
            generated.append(canvas)
            if torch.isin(canvas, pipeline.eos_token_ids).any():
                break
            kv_cache = pipeline.build_kv_cache(canvas[None], kv_cache)  # append finished block

        gen_tokens = pipeline.strip_thinking_tokens(torch.cat(generated))
        return pipeline.processor.decode(gen_tokens, skip_special_tokens=True)

    def run(self):
        self.pipeline.model.eval()
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            self.sampling_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        noise = sample_noise(self.mu, self.sigma, self.N_local, self.accelerator.device)  # (H, N_local, L_noise, d)

        codes = []
        rewards = []
        for b in range(self.N_local):
            text = self.generate(noise[:, b])
            code = self.task.extract_code(text)
            rewards.append(self.task.evaluate_code(code))
            codes.append(code)

        rewards = torch.tensor(rewards, device=self.accelerator.device, dtype=torch.float32)

        # gather along the sample axis (dim 1), so put it first for the gather then restore (H, N, ...)
        gathered_noise   = einops.rearrange(self.accelerator.gather(einops.rearrange(noise, "H N ... -> N H ...")), "N H ... -> H N ...")
        gathered_codes   = gather_object(codes)                # variable-length strings -> object gather
        gathered_rewards = self.accelerator.gather(rewards)

        best_idx = gathered_rewards.argmax().item()
        if gathered_rewards[best_idx].item() > self.best[1]:
            self.best = (gathered_codes[best_idx], gathered_rewards[best_idx].item())

        H, N = gathered_noise.shape[0], gathered_noise.shape[1]
        objective_values = (-gathered_rewards)[None].expand(H, N)  # one reward per sample, shared across layers
        self.mu, self.sigma = update_parameters(self.mu, self.sigma, gathered_noise, objective_values, lr=self.config.lr)
        self.mu = broadcast(self.mu)        # keep parameters identical across ranks
        self.sigma = broadcast(self.sigma)

        # parameter-drift diagnostics (logging only): average per-token L2 norm of mu, total |sigma|.
        mu_l2_norm = self.mu.norm(dim=-1).mean().item()
        sigma_abs_sum = self.sigma.abs().sum().item()

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling", extra={"sampling/best-so-far": self.best[1], "sampling/mu-l2-norm": mu_l2_norm, "sampling/sigma-abs-sum": sigma_abs_sum})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_codes, stage="sampling")


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
