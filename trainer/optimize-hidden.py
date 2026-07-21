import math
import sys

import einops
import torch
from absl import flags
from accelerate.utils import broadcast, gather_object
from ml_collections import config_flags
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from base import BaseTrainer
from mixins import DistributedSubsampleDataset

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/optimize-hidden.py", "Training configuration.")


def project_to_sphere(x):
    """Rescale every (layer, token) hidden vector to the radius-sqrt(D) sphere. rms(x) then equals 1,
    so each layer's input_layernorm becomes the identity (modulo its weight) -- functionally a no-op
    for the KV injection, but it keeps x on a fixed scale so the step size stays interpretable."""
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True) * math.sqrt(x.shape[-1])


class Trainer(BaseTrainer):
    """Hidden-state policy optimization: L hidden-state vectors injected into every transformer layer's KV
    cache become a single differentiable parameter x, trained by GRPO to maximize the task reward. Each
    epoch draws m questions from the dataset and samples k rollouts per question on top of prompt + noise
    KV (one group per batch, the same x injected for all), z-score-normalizes reward within each group into
    advantages, and takes one Adam step on x that raises the advantage-weighted log-likelihood of the
    sampled tokens (single on-policy step: ratio == 1, no PPO clip, no KL)."""

    def __init__(self, config):
        super().__init__(config)

        G = self.accelerator.num_processes
        assert config.sample.total_samples % G == 0, "total_samples must be divisible by num_processes"
        self.N_local = config.sample.total_samples // G

        # Per-layer KV-noise hidden states x: (H, L, D). Seeded off a CPU generator so every rank
        # starts from an identical x -- setup_accelerator's set_seed(..., device_specific=True) would
        # otherwise give each rank a different draw. Matches the torch.Generator idiom in test-noise.py.
        H = self.model.config.num_hidden_layers
        L = self.config.sample.noise_length
        D = self.model.config.hidden_size
        gen = torch.Generator().manual_seed(self.config.seed)
        x = project_to_sphere(torch.randn(H, L, D, generator=gen))
        self.x = x.to(self.accelerator.device, torch.float32).requires_grad_(True)  # fp32 master copy
        self.optimizer = torch.optim.Adam([self.x], lr=self.config.train.learning_rate)

        # GSM-Hard drawn in groups: each batch is one question's k rollouts (identical prompt), so the noise
        # cache injects at a single position and advantages normalize within the group, on one rank.
        self.train_dataset = DistributedSubsampleDataset(
            all_data=self.task.data,
            B=config.sample.total_samples,
            G=G,
            m=config.sample.m,
            b_max=config.sample.total_samples,
            base_seed=config.seed,
        )
        assert self.train_dataset.m % G == 0, "m must be divisible by num_processes (whole groups per rank)"
        training_dataloader = DataLoader(self.train_dataset, batch_size=self.train_dataset.k, shuffle=False)
        self.training_dataloader = self.accelerator.prepare(training_dataloader)

    def build_noisy_cache(self, prompt_tokens):
        # Build the clean prompt KV cache, then inject L "random" KV rows into every layer -- each row
        # produced by pushing this parameter's hidden state x[i] through that layer's real K/V pipeline
        # (input_layernorm -> k/v_proj -> k_norm -> RoPE), landing the keys/values on the model's true
        # K/V manifold (cf. test-noise.py Stage 2). The same noise x is broadcast over the (B,) batch.
        # Differentiable in x; wrap the call in torch.no_grad when the graph is not wanted. Returns
        # (cache, prompt_logits, P).
        L = self.config.sample.noise_length
        B = prompt_tokens.shape[0]
        with torch.no_grad():
            out = self.model(input_ids=prompt_tokens, use_cache=True, logits_to_keep=1)  # only the last row is read
        cache = out.past_key_values
        P = cache.get_seq_length()
        pos = torch.arange(P, P + L, device=self.model.device).unsqueeze(0)  # positions P .. P+L-1
        rotary = self.model.model.rotary_emb
        for i, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            dev = attn.k_proj.weight.device
            Dh = attn.head_dim
            x = self.x[i][None].to(dev, attn.k_proj.weight.dtype).expand(B, -1, -1)  # (B, L, D)
            xln = layer.input_layernorm(x)  # identity modulo weight, since project_to_sphere fixed rms(x) == 1
            k = attn.k_norm(einops.rearrange(attn.k_proj(xln), "b l (h d) -> b l h d", d=Dh))  # k_norm over head_dim
            k = einops.rearrange(k, "b l h d -> b h l d")               # (B, Hkv, L, Dh)
            v = einops.rearrange(attn.v_proj(xln), "b l (h d) -> b h l d", d=Dh)  # (B, Hkv, L, Dh)
            cos, sin = rotary(v, pos.to(dev))
            k, _ = apply_rotary_pos_emb(k, k, cos.to(dev), sin.to(dev))  # RoPE at positions P .. P+L-1
            cl = cache.layers[i]
            cl.keys = torch.cat([cl.keys, k.to(cl.keys.device)], dim=2)
            cl.values = torch.cat([cl.values, v.to(cl.values.device)], dim=2)
        return cache, out.logits, P

    @torch.no_grad()
    def generate(self, prompt_tokens):
        # Stochastic AR decode over prompt + noise KV, batched over the group's k identical prompts. The
        # first token is seeded from the prompt-only logits (it does not attend to the noise: causality
        # forbids position P-1 from seeing rows at >=P); every later token is sampled at position >= P+L and
        # attends over the noise KV. Padding the input to P+L+1 makes `generate` feed only first_token (at
        # position P+L) and take over -- the L placeholders stand in for the injected KV rows and are never
        # embedded (cf. test-noise.py).
        cfg = self.config.sample
        L = cfg.noise_length
        k = prompt_tokens.shape[0]
        cache, prompt_logits, P = self.build_noisy_cache(prompt_tokens)
        first_token = prompt_logits[:, -1:, :].argmax(-1)  # (k, 1), identical within the group
        placeholders = prompt_tokens.new_full((k, L), self.tokenizer.pad_token_id)
        input_ids = torch.cat([prompt_tokens, placeholders, first_token], dim=1)  # (k, P + L + 1)
        attention_mask = torch.ones_like(input_ids)  # all-ones: the placeholders must not read as padding
        out = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        return self.strip_eos(out[:, P + L:])  # k x (T_i,) response tokens y_0 .. y_{T-1}, right-pad dropped

    def run(self):
        self.model.eval()
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            training_data = self.sampling_step(epoch)
            self.training_step(epoch=epoch, training_data=training_data)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.model.eval()
        cfg = self.config.sample

        self.train_dataset.subsample(epoch)
        training_data = []      # one dict per group (question)
        all_responses, all_rewards = [], []
        for data_ids in tqdm(self.training_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            data_id, k = int(data_ids[0]), len(data_ids)
            prompt = self.build_prompt_tokens(self.task.prompt(data_id), system_prompt=self.task.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking)  # (1, P)

            response_tokens = self.generate(prompt.expand(k, -1))  # k x (T_i,), the same noise injected for all k
            responses = self.tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
            rewards = torch.tensor([self.task.evaluate(data_id, r) for r in responses], device=self.accelerator.device, dtype=torch.float32)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)  # group-wise, local (group lives on one rank)

            training_data.append({
                "prompt_tokens": prompt.cpu(),                          # (1, P)
                "response_tokens": [t.cpu() for t in response_tokens],  # k x (T_i,)
                "advantages": advantages.cpu(),                         # (k,)
            })
            all_responses.extend(responses)
            all_rewards.append(rewards)

        rewards = torch.cat(all_rewards)
        gathered_rewards = self.accelerator.gather(rewards)
        gathered_responses = gather_object(all_responses)

        objective_evaluations = epoch * self.config.sample.total_samples
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling")
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_responses, stage="sampling")

        return training_data

    def training_step(self, epoch, training_data):
        n_local = sum(len(group["response_tokens"]) for group in training_data)  # this rank's rollouts (B_i)

        # One Adam step per epoch, on-policy. The gradient is accumulated over this rank's rollouts, then
        # averaged across ranks -- never accumulated across epochs. With a single gradient step the PPO
        # ratio is identically 1, so the surrogate is just -advantage * log_prob; no ratio, no clip, no KL.
        self.optimizer.zero_grad()
        losses = []
        for group in training_data:
            prompt_tokens = group["prompt_tokens"].to(self.model.device)  # (1, P)
            for tokens, advantage in zip(group["response_tokens"], group["advantages"]):
                tokens = tokens.to(self.model.device)
                advantage = advantage.to(self.model.device)
                cache, _, _ = self.build_noisy_cache(prompt_tokens)  # fresh per sample: the forward appends the response K/V to it
                logits = self.model(input_ids=tokens[None], past_key_values=cache, use_cache=True).logits  # (1, T, vocab)
                # logits[:, j] predicts y_{j+1}; y_0 is dropped -- it is seeded from the prompt-only logits
                # and cannot attend to the noise, so its gradient w.r.t. x is exactly zero.
                log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                token_log_probs = log_probs.gather(-1, tokens[None, 1:, None]).squeeze(-1).squeeze(0)  # (T-1,)
                loss = -(advantage * token_log_probs.mean())
                self.accelerator.backward(loss / n_local)  # accumulate; graph freed after each backward
                losses.append(loss.detach())

        # Average the accumulated gradient across ranks (x is a bare tensor, so nothing syncs it for us).
        self.x.grad = self.accelerator.reduce(self.x.grad, reduction="mean")
        grad_norm = self.x.grad.norm(dim=-1).mean()  # L2 over D, averaged over the H*L vectors
        self.optimizer.step()
        with torch.no_grad():
            self.x.copy_(project_to_sphere(self.x))
        self.x.data = broadcast(self.x.data)  # keep x bit-identical across ranks

        with torch.no_grad():
            # Mean inner product between distinct noise rows of a layer, averaged over layers: a
            # collapse probe. Lives in [-D, D] -- ~0 while the rows stay mutually orthogonal, ~D once
            # they have folded onto one direction. The diagonal is exactly D (x is on the sphere), so
            # subtracting it leaves the n*(n-1) ordered off-diagonal pairs.
            gram = self.x @ self.x.mT  # (H, L, L)
            n = gram.shape[-1]
            diag = torch.diagonal(gram, dim1=-2, dim2=-1)  # (H, L)
            pairwise_inner_product = ((gram.sum(dim=(-2, -1)) - diag.sum(-1)) / (n * (n - 1))).mean()

        loss_value = torch.stack(losses).mean()
        gathered_loss = self.accelerator.gather(loss_value.reshape(1)).mean().item()
        objective_evaluations = epoch * self.config.sample.total_samples
        self.accelerator.log({
            "objective-evaluations": objective_evaluations,
            "training/loss": gathered_loss,
            "training/grad-norm": grad_norm.item(),
            "training/x-concentration": pairwise_inner_product.item(),
        })


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
