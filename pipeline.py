import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoProcessor, DiffusionGemmaForBlockDiffusion


def token_entropy(logits):
    """Per-position Shannon entropy H[i] = -sum_v P[i,v] log P[i,v]; (..., V) -> (...,).

    Matches torch.distributions.Categorical(logits=logits).entropy(); computed in
    float32 from log-softmax (stable: softmax probs are never exactly 0, so no -inf).
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(-1)


class DiffusionGemmaScheduler:
    """Timestep schedule + linear temperature for one DiffusionGemma diffusion pass."""

    def __init__(self, t_min=0.4, t_max=0.8):
        self.t_min = t_min
        self.t_max = t_max

    def set_timesteps(self, num_inference_steps, device):
        """
        Args:
            num_inference_steps: DiffusionGemma's num_inference_steps.
        Returns:
            timesteps: (num_inference_steps,) counting down N..1.
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, 0, -1, device=device)
        return self.timesteps

    def temperature(self, logits, timstep):
        """Linear temperature schedule t = t_min + (t_max - t_min) * (timstep / N).

        timstep counts down N..1 (steps remaining), scalar or a per-position (L,) vector.
        Shape-preserving: returns logits / t. A vector timstep is broadcast over the vocab
        dim (unsqueezed to (L, 1)); a scalar / 0-dim timstep is left as-is.
        """
        t = self.t_min + (self.t_max - self.t_min) * (timstep / self.num_inference_steps)
        if torch.is_tensor(t) and t.ndim > 0:
            t = t.unsqueeze(-1)  # (L,) -> (L, 1) to divide (L, V) per position
        return logits / t


class DiffusionGemmaPipeline:
    def __init__(self, model_name, entropy_bound=0.1, confidence_threshold=0.005, t_min=0.4, t_max=0.8, dtype=torch.bfloat16, *, gen_length, tp_plan=None, device_mesh):
        self.processor = AutoProcessor.from_pretrained(model_name)

        # pin the decoder canvas length to gen_length (else it stays at the model default 256)
        config = AutoConfig.from_pretrained(model_name)
        config.canvas_length = gen_length

        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_name, config=config, dtype=dtype, tp_plan=tp_plan, device_mesh=device_mesh)

        # TP group + its root rank: sampling is broadcast from the root so every rank in the
        # group consumes byte-identical tokens (a size-1 group makes the broadcast a no-op).
        self.tp_group = device_mesh["tp"].get_group()
        self.tp_main = dist.get_process_group_ranks(self.tp_group)[0]

        # no image input: drop the vision modules
        del self.model.model.encoder.vision_tower
        del self.model.model.encoder.embed_vision

        self.gen_length = self.model.config.canvas_length
        self.vocab_size = self.model.config.text_config.vocab_size
        self.hidden_size = self.model.config.text_config.hidden_size
        self.device = self.model.get_input_embeddings().weight.device

        self.entropy_bound = entropy_bound
        self.confidence_threshold = confidence_threshold
        self.scheduler = DiffusionGemmaScheduler(t_min=t_min, t_max=t_max)

        self.eos_token_id = torch.tensor(self.model.generation_config.eos_token_id, device=self.device)

    def broadcast_tp(self, tensor):
        """Broadcast from the TP-group root so every rank agrees (no-op when TP size is 1)."""
        dist.broadcast(tensor, src=self.tp_main, group=self.tp_group)
        return tensor

    def sample_logits_to_tokens(self, xt_logits):
        """Entropy-bound sample of the next renoised canvas from logits.

        Accepted positions are sampled from Categorical(P_t); the rest are renoised to
        Uniform(V). Equivalent to the official accept_canvas + renoise_canvas. The result is
        TP-broadcast from the group root, so every rank returns byte-identical tokens.

        Args:
            xt_logits: (L, V) temperature-processed logits.
        Returns:
            xt_tokens_next: (L,) renoised tokens for the next step.
        """
        entropy = token_entropy(xt_logits)  # H_t[i], (L,)

        # Accept the lowest-entropy positions whose preceding entropies sum within the
        # bound: A_t = { the k* lowest-entropy positions : sum_{j<k} H_(j) <= eps }.
        sorted_entropy, order = torch.sort(entropy, descending=False)
        preceding_sum = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy  # sum of strictly-lower entropies
        accept = torch.empty_like(entropy, dtype=torch.bool)
        accept[order] = preceding_sum <= self.entropy_bound  # (L,) acceptance mask A_t

        probs = torch.softmax(xt_logits, dim=-1, dtype=torch.float32)         # P_t, (L, V)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)         # Categorical(P_t[i])
        noise = torch.randint(self.vocab_size, entropy.shape, device=xt_logits.device)  # Uniform(V)
        return self.broadcast_tp(torch.where(accept, sampled, noise))         # x_{t-1}[i], (L,)

    def sample_init_tokens(self):
        """Fully-noised initial canvas: (L,) tokens ~ Uniform(V).

        Equivalent to `sample_logits_to_tokens` on uniform logits (all positions renoised to
        Uniform(V)). Like it, the draw is TP-broadcast from the group root so every rank starts
        the block from byte-identical tokens.
        """
        return self.broadcast_tp(torch.randint(self.vocab_size, (self.gen_length,), device=self.device))  # (L,)

    @torch.no_grad()
    def build_prompt_tokens(self, prompt, enable_thinking=True):
        """Tokenize `prompt` into a batch-1 (1, P) input_ids tensor via the chat template."""
        input_ids = self.processor.apply_chat_template([{"role": "user", "content": prompt}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt",enable_thinking=enable_thinking,)["input_ids"].to(self.device)  # (1, P)
        return input_ids

    @torch.no_grad()
    def build_kv_cache(self, tokens, past_key_values=None):
        """Encode `tokens` (1, T) into the encoder KV cache; returns the cache.

        With `past_key_values=None`, prefill a fresh cache from the prompt. Otherwise append
        `tokens` (e.g. a finished canvas) to the existing cache: the encoder self-attention
        calls `past_key_values.update(...)`, and with position ids left unset places the new
        tokens in the next T slots (`arange(T) + past_key_values.get_seq_length()`). Only the
        encoder ever writes the cache -- the decoder reads it -- so this is how the official
        `generate` grows the cache one canvas per block.
        """
        out = self.model.model.encoder(input_ids=tokens, past_key_values=past_key_values)
        return out.past_key_values

    def early_stop(self, xt_logits, xt_logits_next):
        """Per-position stopping mask: a position is finished when it is stable and confident.

        The result is TP-broadcast from the group root so every rank's loop length stays in
        lockstep (guards against float-level disagreement flipping `finished[-1]` on one rank).

        - stable[i]: the argmax (clean) prediction at position i is unchanged from the
          previous step's logits.
        - confident[i]: the left-to-right cumulative mean token entropy over positions
          [0..i] is below `self.confidence_threshold` (so confident[-1] uses the whole canvas).

        Args:
            xt_logits: (L, V) previous step's temperature-processed logits, or None on step 0.
            xt_logits_next: (L, V) current step's temperature-processed logits.
        Returns:
            finished: (L,) bool mask; the caller checks `.all()` to stop the diffusion.
        """
        if xt_logits is None:
            return torch.zeros(xt_logits_next.shape[0], device=xt_logits_next.device, dtype=torch.bool)  # (L,)

        stable = torch.argmax(xt_logits, dim=-1) == torch.argmax(xt_logits_next, dim=-1)  # (L,)
        entropy = token_entropy(xt_logits_next)  # (L,)
        cum_mean = torch.cumsum(entropy, dim=-1) / torch.arange(1, entropy.shape[-1] + 1, device=entropy.device)
        confident = cum_mean < self.confidence_threshold  # (L,)
        return self.broadcast_tp(stable & confident)  # (L,)

    @torch.no_grad()
    def model_predict(self, xt_tokens, xt_logits, timesteps, kv_cache, output_hidden_states=True):
        """One denoiser step. The caller owns sampling (`xt_tokens`); stopping is `self.early_stop`.

        Args:
            xt_tokens: (1, L) renoised canvas tokens for this step (from the pipeline's
                sampling methods, which TP-broadcast so every rank passes identical tokens).
            xt_logits: (L, V) previous step's temperature-processed logits, used as the
                self-conditioning signal, or None on the first step (self-conditioning off).
            timesteps: per-position "time lives", counting down N..1; a (L,) vector or a
                scalar (broadcast).
            kv_cache: batch-1 encoder KV cache from `build_kv_cache`.
        Returns:
            xt_logits_next: (L, V) temperature-processed logits.
            hidden_states: (H+1, L, D) per-layer hidden states.
        """
        # decoder canvas sits right after the prompt: positions P .. P+L-1
        P = kv_cache.get_seq_length()
        L = xt_tokens.shape[-1]

        out = self.model(
            input_ids=None,
            past_key_values=kv_cache,
            decoder_position_ids=torch.arange(P, P + L, device=self.device).unsqueeze(0),
            decoder_input_ids=xt_tokens,
            self_conditioning_logits=xt_logits,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = torch.stack(out.hidden_states, dim=0)[:, 0]  # (H+1, L, D)
        xt_logits_next = self.scheduler.temperature(out.logits[0], timstep=timesteps)  # (L, V)

        finished = self.early_stop(xt_logits, xt_logits_next)  # (L,)

        return xt_logits_next, hidden_states, finished

    def argmax_logits_to_tokens(self, logits):
        """Select argmax logits as the tokens"""
        return torch.argmax(logits, dim=-1)  # (L,)

    def argmax_logits_to_text(self, logits, skip_special_tokens=True):
        """Decode a single (T,) token sequence into a string."""
        tokens = self.argmax_logits_to_tokens(logits)  # (L,)
        texts = self.processor.decode(tokens, skip_special_tokens=skip_special_tokens)
        return texts

    def strip_thinking_tokens(self, tokens):
        """Drop everything up to and including the last <channel|> (eoc) token.
        """
        eoc_id = self.processor.tokenizer.eoc_token_id
        positions = (tokens == eoc_id).nonzero()
        if len(positions) == 0:
            return tokens
        return tokens[positions[-1].item() + 1:]
