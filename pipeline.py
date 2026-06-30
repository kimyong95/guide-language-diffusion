import torch
from transformers import AutoConfig, AutoProcessor, DiffusionGemmaForBlockDiffusion
from gemma_utils import build_device_map


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

        timstep counts down N..1 (steps remaining). Shape-preserving: returns logits / t.
        """
        t = self.t_min + (self.t_max - self.t_min) * (timstep / self.num_inference_steps)
        return logits / t


class DiffusionGemmaPipeline:
    def __init__(self, model_name, entropy_bound=0.1, confidence_threshold=0.005, t_min=0.4, t_max=0.8, dtype=torch.bfloat16, *, gen_length):
        self.processor = AutoProcessor.from_pretrained(model_name)

        # pin the decoder canvas length to gen_length (else it stays at the model default 256)
        config = AutoConfig.from_pretrained(model_name)
        config.canvas_length = gen_length

        if model_name == "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic":
            from extend_diffusion_gemma_fp8 import patch_diffusion_gemma_fp8
            patch_diffusion_gemma_fp8()

        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_name, config=config, dtype=dtype, device_map="cuda")

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

        # encoder KV cache + decoder positions, set by set_prompt
        self.kv_cache = None
        self.dec_pos = None

    def sample_logits_to_tokens(self, xt_logits):
        """Entropy-bound sample of the next renoised canvas from logits.

        Accepted positions are sampled from Categorical(P_t); the rest are renoised to
        Uniform(V). Equivalent to the official accept_canvas + renoise_canvas.

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
        return torch.where(accept, sampled, noise)                           # x_{t-1}[i], (L,)
    
    def early_stop(self, xt_logits, xt_logits_next):
        """Stateless stopping check: stop when the clean prediction is stable and confident.

        - stable: the argmax (clean) prediction is unchanged from the previous step's logits.
        - confident: the mean token entropy of the current logits is below `confidence_threshold`.

        Args:
            xt_logits: (L, V) previous step's temperature-processed logits.
            xt_logits_next: (L, V) current step's temperature-processed logits.
        Returns:
            finished: bool, True if the diffusion should stop.
        """
        stable = bool((torch.argmax(xt_logits, dim=-1) == torch.argmax(xt_logits_next, dim=-1)).all())
        confident = bool(token_entropy(xt_logits_next).mean() < self.confidence_threshold)
        return stable and confident

    @torch.no_grad()
    def set_prompt(self, prompt):
        """Encode `prompt` into the batch-1 KV cache (`self.kv_cache`) and set `self.dec_pos`."""
        input_ids = self.processor.apply_chat_template([{"role": "user", "content": prompt}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt",enable_thinking=False,)["input_ids"].to(self.device)  # (1, P)
        P = input_ids.shape[1]
        L = self.gen_length
        self.dec_pos = torch.arange(P, P + L, device=self.device).unsqueeze(0)  # (1, L)
        out = self.model.model.encoder(input_ids=input_ids)
        self.kv_cache = out.past_key_values

    @torch.no_grad()
    def model_predict_step(self, xt_logits, timestep, output_hidden_states=True):
        """
        Args:
            xt_logits: (L, V) previous step's temperature-processed logits, or None on step 0.
            timestep: scalar timestep, counting down N..1.
        Returns:
            xt_logits_next: (L, V) temperature-processed logits.
            hidden_states: (H+1, L, D) per-layer hidden states.
            early_stop: bool from the stopping criteria (False on step 0).
        """
        
        xt_tokens = self.sample_logits_to_tokens(xt_logits)[None]

        out = self.model(
            input_ids=None,
            past_key_values=self.kv_cache,
            decoder_position_ids=self.dec_pos,
            decoder_input_ids=xt_tokens,
            self_conditioning_logits=xt_logits if timestep != self.scheduler.timesteps[0] else None,  # None on step 0
            output_hidden_states=output_hidden_states,
        )
        hidden_states = torch.stack(out.hidden_states, dim=0)[:, 0]  # (H+1, L, D)
        xt_logits_next = self.scheduler.temperature(out.logits, timstep=timestep)[0]  # (L, V)

        early_stop = self.early_stop(xt_logits, xt_logits_next)

        return xt_logits_next, hidden_states, early_stop

    def argmax_logits_to_tokens(self, logits):
        """Select argmax logits as the tokens"""
        return torch.argmax(logits, dim=-1)  # (L,)

    def argmax_logits_to_text(self, logits, skip_special_tokens=True):
        """Decode a single (T,) token sequence into a string."""
        tokens = self.argmax_logits_to_tokens(logits)  # (L,)
        texts = self.processor.decode(tokens, skip_special_tokens=skip_special_tokens)
        return texts
