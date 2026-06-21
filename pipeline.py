import torch
from transformers import AutoConfig, AutoProcessor, DiffusionGemmaForBlockDiffusion
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    EntropyBoundSampler,
    EntropyBoundSamplerConfig,
    LinearTemperatureScheduleLogitsProcessor,
    StableAndConfidentStoppingCriteria,
)
from gemma_utils import build_device_map

class DiffusionGemmaScheduler:
    """Stateless scheduler for one DiffusionGemma diffusion pass over a single sequence.

    xt_tokens: (L,) long; xt_logits: (L, V) float.
    """

    def __init__(
        self,
        vocab_size,
        gen_length=256,
        entropy_bound=0.1,
        t_min=0.4,
        t_max=0.8,
        stability_threshold=1,
        confidence_threshold=0.005,
    ):
        self.gen_length = gen_length
        self.t_min = t_min
        self.t_max = t_max

        self.sampler = EntropyBoundSampler(
            config=EntropyBoundSamplerConfig(entropy_bound=entropy_bound),
            canvas_length=gen_length,
            vocab_size=vocab_size,
            max_denoising_steps=0,  # unused by this sampler
        )
        self.stopping_criteria = StableAndConfidentStoppingCriteria(
            stability_threshold=stability_threshold,
            confidence_threshold=confidence_threshold,
        )

    def set_timesteps(self, num_inference_steps, device):
        """
        Args:
            num_inference_steps: DiffusionGemma's max_denoising_steps.
        Returns:
            timesteps: (num_inference_steps,) counting down N..1.
        """
        self.max_denoising_steps = num_inference_steps
        self.temperature = LinearTemperatureScheduleLogitsProcessor(
            t_min=self.t_min, t_max=self.t_max, max_denoising_steps=num_inference_steps
        )
        self.timesteps = torch.arange(num_inference_steps, 0, -1, device=device)
        return self.timesteps

    def init_tokens(self, device):
        """Returns: (L,) random initial tokens; resets the stopping criteria."""
        self.stopping_criteria.reset()
        return self.sampler.initialize_canvas(batch_size=1, device=device)[0]  # (L,)

    def step(self, xt_tokens, xt_logits, timestep):
        """
        Args:
            xt_tokens: (L,) long working tokens.
            xt_logits: (L, V) temperature-processed logits.
        Returns:
            next_xt_tokens: (L,) renoised tokens for the next step.
            x1_tokens: (L,) clean prediction.
            finished: scalar bool from the stopping criteria.
        """
        # sampler / stopping utilities are batched; add a temporary batch-1 dim
        xt_tokens_b = xt_tokens[None]  # (1, L)
        xt_logits_b = xt_logits[None]  # (1, L, V)

        probs = torch.softmax(xt_logits, dim=-1, dtype=torch.float32)          # (L, V)
        denoiser_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (L,)
        x1_tokens = torch.argmax(xt_logits, dim=-1)                            # (L,)

        accepted_tokens = self.sampler.accept_canvas(xt_tokens_b, denoiser_tokens[None], xt_logits_b, timestep)
        next_xt_tokens = self.sampler.renoise_canvas(accepted_tokens, timestep)[0]  # (L,)

        finished = self.stopping_criteria(x1_tokens[None], xt_logits_b)[0]  # scalar bool

        return next_xt_tokens, x1_tokens, finished


class DiffusionGemmaPipeline:
    def __init__(self, model_name, entropy_bound=0.1, t_min=0.4, t_max=0.8, dtype=torch.bfloat16, *, gen_length):
        self.processor = AutoProcessor.from_pretrained(model_name)

        # pin the decoder canvas length to gen_length (else it stays at the model default 256)
        config = AutoConfig.from_pretrained(model_name)
        config.canvas_length = gen_length
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_name, config=config, dtype=dtype, device_map=build_device_map())

        # no image input: drop the vision modules
        del self.model.model.encoder.vision_tower
        del self.model.model.encoder.embed_vision

        self.gen_length = self.model.config.canvas_length
        self.vocab_size = self.model.config.text_config.vocab_size
        self.hidden_size = self.model.config.text_config.hidden_size
        self.input_device = self.model.get_input_embeddings().weight.device

        self.scheduler = DiffusionGemmaScheduler(
            vocab_size=self.vocab_size,
            gen_length=self.gen_length,
            entropy_bound=entropy_bound,
            t_min=t_min,
            t_max=t_max,
        )

        # encoder KV cache + decoder positions, set by set_prompt
        self.kv_cache = None
        self.dec_pos = None

    def init_tokens(self):
        """Returns: (L,) random initial tokens; resets the stopping criteria."""
        random_tokens = self.scheduler.init_tokens(self.input_device)
        return random_tokens

    @torch.no_grad()
    def set_prompt(self, prompt):
        """Encode `prompt` into the batch-1 KV cache (`self.kv_cache`) and set `self.dec_pos`."""
        input_ids = self.processor.apply_chat_template([{"role": "user", "content": prompt}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt",enable_thinking=False,)["input_ids"].to(self.input_device)  # (1, P)
        P = input_ids.shape[1]
        L = self.gen_length
        self.dec_pos = torch.arange(P, P + L, device=self.input_device).unsqueeze(0)  # (1, L)
        out = self.model.model.encoder(input_ids=input_ids)
        self.kv_cache = out.past_key_values

    @torch.no_grad()
    def model_predict(self, xt_tokens, self_conditioning_logits, timestep, output_hidden_states=True):
        """
        Args:
            xt_tokens: (L,) long.
            self_conditioning_logits: (L, V) from the previous step, or None.
        Returns:
            xt_logits: (L, V) temperature-processed logits.
            hidden_states: (H+1, L, D) per-layer hidden states.
        """
        out = self.model(
            input_ids=None,
            past_key_values=self.kv_cache,
            decoder_input_ids=xt_tokens[None],  # (1, L)
            decoder_position_ids=self.dec_pos,
            self_conditioning_logits=self_conditioning_logits,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = torch.stack(out.hidden_states, dim=0)[:, 0]  # (H+1, L, D)
        xt_logits = self.scheduler.temperature(None, out.logits, cur_step=timestep)[0]  # (L, V)
        return xt_logits, hidden_states

    def tokens_to_text(self, tokens, skip_special_tokens=False):
        """Decode a single (T,) token sequence into a string."""
        return self.processor.decode(tokens, skip_special_tokens=skip_special_tokens)
