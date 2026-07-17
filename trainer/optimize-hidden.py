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
config_flags.DEFINE_config_file("config", "config/optimize-hidden.py", "Training configuration.")


def project_to_sphere(x):
    """Rescale every (layer, token) hidden vector to the radius-sqrt(D) sphere. rms(x) then equals 1,
    so each layer's input_layernorm becomes the identity (modulo its weight) -- functionally a no-op
    for the KV injection, but it keeps x on a fixed scale so the SGD step size stays interpretable."""
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True) * math.sqrt(x.shape[-1])


class Trainer(BaseTrainer):
    """Hidden-state policy optimization: the per-layer random KV rows of test-qwen-noise.py Stage 2 --
    L hidden-state vectors injected into every transformer layer's KV cache -- become a single
    differentiable parameter x, trained by policy gradient (GRPO, reduced to plain PPO with one fixed
    prompt) to maximize the task reward. Each epoch samples N stochastic rollouts on top of prompt +
    noise KV, evaluates each, then takes one SGD step on x that raises the log-likelihood of the
    sampled tokens weighted by their advantage. The epoch / N-per-epoch loop mirrors best-of-n; the
    KV injection mirrors optimize-noise, but x is optimized by gradient rather than an ES update."""

    def __init__(self, config):
        super().__init__(config)

        N = self.config.sample.total_samples
        G = self.accelerator.num_processes
        assert N % G == 0, "total_samples must be divisible by num_processes"
        self.N_local = N // G

        # Per-layer KV-noise hidden states x: (H, L, D). Seeded off a CPU generator so every rank
        # starts from an identical x -- setup_accelerator's set_seed(..., device_specific=True) would
        # otherwise give each rank a different draw. Matches the torch.Generator idiom in test-noise.py.
        H = self.model.config.num_hidden_layers
        L = self.config.sample.noise_length
        D = self.model.config.hidden_size
        gen = torch.Generator().manual_seed(self.config.seed)
        x = project_to_sphere(torch.randn(H, L, D, generator=gen))
        self.x = x.to(self.accelerator.device, torch.float32).requires_grad_(True)  # fp32 master copy

        self.optimizer = torch.optim.SGD([self.x], lr=self.config.train.learning_rate)

        # Fixed prompt, encoded once so x optimizes against a stationary objective (cf. optimize-noise):
        # task.evaluate still ratchets its own best code, but that never feeds back into the prompt.
        system_prompt, user_prompt = self.task.prompt()
        self.prompt_tokens = self.build_prompt_tokens(user_prompt, system_prompt=system_prompt, enable_thinking=self.config.sample.enable_thinking)  # (1, P)

    def build_noisy_cache(self):
        # Build the clean prompt KV cache, then inject L "random" KV rows into every layer -- each row
        # produced by pushing this parameter's hidden state x[i] through that layer's real K/V pipeline
        # (input_layernorm -> k/v_proj -> k_norm -> RoPE), landing the keys/values on the model's true
        # K/V manifold (cf. test-noise.py Stage 2). Differentiable in x; wrap the call in torch.no_grad
        # when the graph is not wanted. Returns (cache, prompt_logits, P).
        L = self.config.sample.noise_length
        with torch.no_grad():
            out = self.model(input_ids=self.prompt_tokens, use_cache=True, logits_to_keep=1)  # only the last row is read
        cache = out.past_key_values
        P = cache.get_seq_length()
        pos = torch.arange(P, P + L, device=self.model.device).unsqueeze(0)  # positions P .. P+L-1
        rotary = self.model.model.rotary_emb
        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            dev = attn.k_proj.weight.device
            Dh = attn.head_dim
            x = self.x[i][None].to(dev, attn.k_proj.weight.dtype)  # (1, L, D)
            xln = layer.input_layernorm(x)  # identity modulo weight, since project_to_sphere fixed rms(x) == 1
            k = attn.k_norm(einops.rearrange(attn.k_proj(xln), "b l (h d) -> b l h d", d=Dh))  # k_norm over head_dim
            k = einops.rearrange(k, "b l h d -> b h l d")               # (1, Hkv, L, Dh)
            v = einops.rearrange(attn.v_proj(xln), "b l (h d) -> b h l d", d=Dh)  # (1, Hkv, L, Dh)
            cos, sin = rotary(v, pos.to(dev))
            k, _ = apply_rotary_pos_emb(k, k, cos.to(dev), sin.to(dev))  # RoPE at positions P .. P+L-1
            cl = cache.layers[i]
            cl.keys = torch.cat([cl.keys, k.to(cl.keys.device)], dim=2)
            cl.values = torch.cat([cl.values, v.to(cl.values.device)], dim=2)
        return cache, out.logits, P

    @torch.no_grad()
    def generate(self):
        # Stochastic AR decode over prompt + noise KV. The first token is seeded from the prompt-only
        # logits (it does not attend to the noise: causality forbids position P-1 from seeing rows at
        # >=P); every later token is sampled at position >= P+L and attends over the noise KV. Padding
        # the input to P+L+1 makes `generate` feed only first_token (at position P+L) and take over --
        # the L placeholders stand in for the injected KV rows and are never embedded (cf. test-noise.py).
        L = self.config.sample.noise_length
        cache, prompt_logits, P = self.build_noisy_cache()
        first_token = prompt_logits[:, -1:, :].argmax(-1)  # (1, 1)
        placeholders = self.prompt_tokens.new_full((1, L), self.tokenizer.pad_token_id)
        input_ids = torch.cat([self.prompt_tokens, placeholders, first_token], dim=1)  # (1, P + L + 1)
        attention_mask = torch.ones_like(input_ids)  # all-ones: the placeholders must not read as padding
        out = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            max_new_tokens=self.config.sample.max_new_tokens,
            do_sample=True,
            temperature=self.config.sample.temperature,
            top_p=self.config.sample.top_p,
        )
        return out[0, P + L:]  # (T,) response tokens y_0 .. y_{T-1}

    def run(self):
        self.model.eval()
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            training_data = self.sampling_step(epoch)
            self.training_step(epoch=epoch, training_data=training_data)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        responses = []
        rewards = []
        response_tokens = []
        for _ in range(self.N_local):
            tokens = self.generate()
            response = self.tokenizer.decode(tokens, skip_special_tokens=True)
            rewards.append(self.task.evaluate(response))  # ratchets the task's best code across ranks
            responses.append(response)
            response_tokens.append(tokens.cpu())  # variable-length long tensor per sample

        rewards = torch.tensor(rewards, device=self.accelerator.device, dtype=torch.float32)

        gathered_responses = gather_object(responses)          # variable-length strings -> object gather
        gathered_rewards = self.accelerator.gather(rewards)

        # GRPO advantages: one fixed prompt means one group, so the group is the whole gathered batch
        # and this reduces to plain batch normalization. Scatter back to this rank's slice.
        advantages = (gathered_rewards - gathered_rewards.mean()) / gathered_rewards.std().clamp(min=1e-8)
        advantages = einops.rearrange(advantages, "(process batch) -> process batch", process=self.accelerator.num_processes)[self.accelerator.process_index]

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_responses, stage="sampling")

        return {"response_tokens": response_tokens, "advantages": advantages}

    def training_step(self, epoch, training_data):
        response_tokens = training_data["response_tokens"]
        advantages = training_data["advantages"]
        adv_clip_max = self.config.train.adv_clip_max

        # One SGD step per epoch, on-policy. The gradient is accumulated over this rank's tokens and
        # samples, then averaged across ranks -- never accumulated across epochs. With a single
        # gradient step the PPO ratio is identically 1, so the surrogate is just -advantage * log_prob;
        # no ratio, no clipping, no KL.
        self.optimizer.zero_grad()
        losses = []
        for tokens, advantage in zip(response_tokens, advantages):
            tokens = tokens.to(self.model.device)
            cache, _, _ = self.build_noisy_cache()  # fresh per sample: the forward appends the response K/V to it
            logits = self.model(input_ids=tokens[None], past_key_values=cache, use_cache=True).logits  # (1, T, vocab)
            # logits[:, j] predicts y_{j+1}; y_0 is dropped -- it is seeded from the prompt-only logits
            # and cannot attend to the noise, so its gradient w.r.t. x is exactly zero.
            log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            token_log_probs = log_probs.gather(-1, tokens[None, 1:, None]).squeeze(-1).squeeze(0)  # (T-1,)
            advantage = advantage.clamp(-adv_clip_max, adv_clip_max)
            loss = -(advantage * token_log_probs.mean())
            self.accelerator.backward(loss / self.N_local)  # accumulate; graph freed after each backward
            losses.append(loss.detach())

        # Average the accumulated gradient across ranks (x is a bare tensor, so nothing syncs it for us).
        self.x.grad = self.accelerator.reduce(self.x.grad, reduction="mean")
        grad_norm = torch.nn.utils.clip_grad_norm_([self.x], self.config.train.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            self.x.copy_(project_to_sphere(self.x))
        self.x.data = broadcast(self.x.data)  # keep x bit-identical across ranks

        loss_value = torch.stack(losses).mean()
        gathered_loss = self.accelerator.gather(loss_value.reshape(1)).mean().item()
        objective_evaluations = epoch * self.config.sample.total_samples
        self.accelerator.log({
            "objective-evaluations": objective_evaluations,
            "training/loss": gathered_loss,
            "training/grad-norm": grad_norm.item(),
            "training/x-l2-norm": self.x.norm(dim=-1).mean().item(),
        })


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
