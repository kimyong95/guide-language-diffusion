import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MASK_ID = 126336  # [MASK] token id for LLaDA-8B


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling. temperature=0 → greedy argmax."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

class LLaDAScheduler:
    """
    Diffusers-style full-diffusion scheduler. Each timestep commits the
    most-confident masked positions over the entire gen sequence; no block
    structure.

    xt is the committed token state, shape (B, L) long;
    logits are the prediction, shape (B, L, V); L = gen_length (no prompt).
    """
    def __init__(self, gen_length=128, temperature=1.0, mask_id=MASK_ID):
        self.gen_length = gen_length
        self.temperature = temperature
        self.mask_id = mask_id

    def set_timesteps(self, num_inference_steps, device):
        base = self.gen_length // num_inference_steps
        rem  = self.gen_length %  num_inference_steps
        self.transfer_per_step = [base + (1 if i < rem else 0) for i in range(num_inference_steps)]
        self.timesteps = torch.arange(num_inference_steps, device=device)
        return self.timesteps

    def get_coefficients(self, timestep):
        return {"num_transfer": self.transfer_per_step[int(timestep)], "t": timestep / len(self.timesteps)}


    def step(self, xt, pred_logits, timestep):
        """One denoising step: Gumbel-max sample from `logits` and commit the
        top-k most-confident sampled tokens into `xt`.

        Args:
            xt (B, L): current committed tokens (mask_id at uncommitted positions), dtype long.
            logits (B, L, V): predicted logits.
            timestep (int | scalar tensor): step index in [0, num_inference_steps).

        Returns:
            xt (B, L): updated committed tokens (in-place; same tensor object), dtype long.
        """
        num_transfer = self.get_coefficients(timestep)["num_transfer"]

        pred_logits = add_gumbel_noise(pred_logits, self.temperature)
        pred_tokens = pred_logits.argmax(dim=-1) # (B, L)
        p = F.softmax(pred_logits.to(torch.float64), dim=-1)
        conf = torch.gather(p, -1, pred_tokens.unsqueeze(-1)).squeeze(-1) # (B, L)

        for i in range(xt.shape[0]):
            masked_pos = torch.where(xt[i] == self.mask_id)[0]
            select = masked_pos[conf[i, masked_pos].topk(num_transfer).indices]
            xt[i, select] = pred_tokens[i, select]

        return xt

class LLaDAPipeline:
    def __init__(self, model_name, device, gen_length, temperature, dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.transformer = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.mask_id = MASK_ID
        self.vocab_size = self.transformer.config.vocab_size
        self.hidden_size = self.transformer.config.hidden_size
        self.scheduler = LLaDAScheduler(gen_length=gen_length, temperature=temperature)

    def encode_prompt(self, question):
        messages = [{"role": "user", "content": question}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(text)['input_ids']
        return torch.tensor(ids, device=self.device).unsqueeze(0)

    def init_tokens(self, batch_size):
        """Initialize the sampling state with all mask tokens.

        Args:
            batch_size (int): number of samples in the batch.

        Returns:
            xt (B, L): mask_id at every position, dtype long.
        """
        return torch.full((batch_size, self.scheduler.gen_length), self.mask_id, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict_latents(self, xt, question):
        """One transformer forward pass over `[prompt | xt]`; return gen-position hidden states.

        Args:
            xt (B, L): current committed tokens (mask_id at uncommitted positions), dtype long.
            question (str): the task prompt; re-encoded internally each call.

        Returns:
            latents (B, L, H): last-layer hidden state at the gen positions, dtype bfloat16.
        """
        qes_tokens = self.encode_prompt(question).expand(xt.shape[0], -1)
        all_tokens = torch.cat([qes_tokens, xt], dim=1)
        P = qes_tokens.shape[1]
        out = self.transformer(all_tokens, output_hidden_states=True)
        return out.hidden_states[-1][:, P:, :]

    def decode(self, xt):
        """Decode committed tokens to text strings via the tokenizer.

        Args:
            xt (B, L): committed tokens, dtype long.

        Returns:
            texts (list[str], length B): one decoded string per batch element,
                with special tokens removed.
        """
        return self.tokenizer.batch_decode(xt, skip_special_tokens=True)


def retrieve_timesteps(scheduler, num_inference_steps, device=None):
    timesteps = scheduler.set_timesteps(num_inference_steps, device)
    return timesteps, num_inference_steps
