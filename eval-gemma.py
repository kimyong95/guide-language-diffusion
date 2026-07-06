'''
lm-evaluation-harness adapter(s) for the DiffusionGemma block-diffusion model.

Two sampling recipes are exposed as separate models sharing one base class; they differ only
in the generation procedure (`generate_tokens`):
  - "diffusion-gemma"         : block diffusion -- denoise a full gen_length canvas per block,
                                argmax-commit the whole block, grow the KV cache, repeat until
                                EOS / max_blocks.
  - "diffusion-gemma-sliding" : sliding window -- per-position timesteps, commit only the
                                leading all-finished prefix each step and slide the window
                                forward by that many fresh positions, until EOS / max_tokens.

Both implement `generate_until` only. Likelihood-based requests (`loglikelihood` /
`loglikelihood_rolling`) are not applicable to a diffusion model and raise NotImplementedError,
so these adapters work with generative tasks (e.g. gsm8k) that grade on the answer string.

The generation recipes are lifted from test-parallel.py / test-sampling.py, with the
tensor-parallel scaffolding (device_mesh, tp_plan, dist.broadcast) stripped away. Data
parallelism is supported: launch with `accelerate launch --num_processes N eval-gemma.py ...`
and each process holds a full copy of the model on its own GPU while lm-eval shards the eval
docs across ranks and gathers the per-sample results back to rank 0 for aggregation.
'''
import random

import numpy as np
import torch
from accelerate import Accelerator
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from pipeline import DiffusionGemmaPipeline


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DiffusionGemmaEvalHarness(LM):
    '''Shared base: model loading, DP plumbing, and the generate_until driver loop.

    Subclasses implement `generate_tokens`, the only per-recipe difference.
    '''

    def __init__(
        self,
        model_name="google/diffusiongemma-26B-A4B-it",
        gen_length=256,
        num_inference_steps=48,
        entropy_bound=0.1,
        confidence_threshold=0.005,
        t_min=0.4,
        t_max=0.8,
        enable_thinking=True,
        **kwargs,
    ):
        '''
        Args:
            model_name: DiffusionGemma checkpoint (instruction-tuned).
            gen_length: decoder canvas length, i.e. tokens produced per block / window.
            num_inference_steps: diffusion denoising steps per position.
            entropy_bound / confidence_threshold / t_min / t_max: sampling schedule
                parameters forwarded to DiffusionGemmaPipeline.
            enable_thinking: apply the chat template with the model's thinking mode on.
        '''
        super().__init__()

        # Always launched with `accelerate launch`: each process holds a full model copy and
        # lm-eval shards docs across ranks (rank/world_size) then gathers results to rank 0.
        self.accelerator = Accelerator()
        # Pin this process to its GPU so the pipeline's device_map="cuda" (the *current* CUDA
        # device) loads on cuda:local_rank instead of every rank piling onto cuda:0.
        torch.cuda.set_device(self.accelerator.local_process_index)

        # lm-eval passes --model_args as strings; cast defensively.
        gen_length = int(gen_length)
        self.num_inference_steps = int(num_inference_steps)
        self.enable_thinking = enable_thinking

        self.pipeline = DiffusionGemmaPipeline(
            model_name,
            entropy_bound=float(entropy_bound),
            confidence_threshold=float(confidence_threshold),
            t_min=float(t_min),
            t_max=float(t_max),
            gen_length=gen_length,
            device_map="cuda",
        )
        self.pipeline.model.eval()

        # set_timesteps also records scheduler.num_inference_steps, which temperature() needs.
        self.timesteps = self.pipeline.scheduler.set_timesteps(self.num_inference_steps, device=self.pipeline.device)

    def loglikelihood(self, requests):
        raise NotImplementedError("DiffusionGemma is a diffusion model; loglikelihood is not supported.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("DiffusionGemma is a diffusion model; loglikelihood_rolling is not supported.")

    # --- data-parallel plumbing: lm-eval reads these to shard docs and gather results ---
    # (the evaluator only calls all_gather/gather_object/barrier when world_size > 1)
    @property
    def rank(self):
        return self.accelerator.local_process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes

    @property
    def device(self):
        return self.pipeline.device

    def all_gather(self, tensor):
        return self.accelerator.gather(tensor)

    def gather_object(self, obj, dst=0):
        result = [None] * self.world_size if self.rank == dst else None
        torch.distributed.gather_object(obj=obj, object_gather_list=result, dst=dst)
        return result

    def barrier(self):
        self.accelerator.wait_for_everyone()

    def generate(self, prompt: str) -> str:
        '''Generate one response, text in / text out. Overridden per sampling recipe.'''
        raise NotImplementedError

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        return [self.generate(req.args[0]) for req in tqdm(requests, desc="Generating...")]


@register_model("diffusion-gemma")
class DiffusionGemmaBlock(DiffusionGemmaEvalHarness):
    '''Block diffusion: denoise the whole canvas per block, argmax-commit it, repeat.'''

    def __init__(self, max_blocks=16, **kwargs):
        '''max_blocks: upper bound on blocks; generation stops early on EOS.'''
        super().__init__(**kwargs)
        self.max_blocks = int(max_blocks)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        pipeline = self.pipeline
        prompt_tokens = pipeline.build_prompt_tokens(prompt, enable_thinking=self.enable_thinking)
        kv_cache = pipeline.build_kv_cache(prompt_tokens)

        generated = []
        for _ in range(self.max_blocks):
            xt_logits = None
            xt_tokens = pipeline.sample_init_tokens()[None]
            for timestep in self.timesteps:
                xt_logits, _, finished = pipeline.model_predict(xt_tokens, xt_logits, timestep, kv_cache)
                xt_tokens = pipeline.sample_logits_to_tokens(xt_logits)[None]
                if finished[-1]:
                    break

            canvas = pipeline.argmax_logits_to_tokens(xt_logits)
            generated.append(canvas)
            if torch.isin(canvas, pipeline.eos_token_id).any():
                break
            kv_cache = pipeline.build_kv_cache(canvas[None], kv_cache)

        gen_tokens = pipeline.strip_thinking_tokens(torch.cat(generated))
        gen_texts = pipeline.processor.decode(gen_tokens, skip_special_tokens=True)

        return gen_texts


@register_model("diffusion-gemma-sliding")
class DiffusionGemmaSliding(DiffusionGemmaEvalHarness):
    '''Sliding window: commit the leading all-finished prefix each step and slide forward.'''

    def __init__(self, max_tokens=4096, **kwargs):
        '''max_tokens: upper bound on committed tokens; generation stops early on EOS.'''
        super().__init__(**kwargs)
        self.max_tokens = int(max_tokens)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        pipeline = self.pipeline
        prompt_tokens = pipeline.build_prompt_tokens(prompt, enable_thinking=self.enable_thinking)
        kv_cache = pipeline.build_kv_cache(prompt_tokens)
        L, V, device = pipeline.gen_length, pipeline.vocab_size, self.device
        N = self.num_inference_steps

        timesteps = torch.full((L,), N, device=device, dtype=torch.long)  # per-position lives
        xt_logits = None                                 # self-conditioning off on the first step
        xt_tokens = pipeline.sample_init_tokens()[None]  # (1, L) fully-noised canvas ~ Uniform(V)

        committed = []       # list of (k,) committed token-id tensors
        n_committed = 0
        while n_committed < self.max_tokens:
            xt_logits, _, finished = pipeline.model_predict(xt_tokens, xt_logits, timesteps, kv_cache)  # (L, V), (L,)
            timesteps = torch.clamp(timesteps - 1, min=0)  # age every position by one step, floor at 0
            finished = finished | (timesteps == 0)

            k = int(finished.long().cumprod(dim=0).sum())  # length of the leading all-finished prefix

            if k > 0:
                commit = pipeline.argmax_logits_to_tokens(xt_logits[:k])  # (k,) clean tokens
                committed.append(commit)
                n_committed += k
                if torch.isin(commit, pipeline.eos_token_id).any():
                    break                                    # EOS reached: stop before caching this commit
                kv_cache = pipeline.build_kv_cache(commit[None], kv_cache)  # grow cache by k -> decoder positions auto-slide

                # slide the window: drop the k committed positions, append k fresh hot positions
                xt_logits = torch.cat([xt_logits[k:], torch.ones(k, V, device=device, dtype=xt_logits.dtype)], dim=0)
                timesteps = torch.cat([timesteps[k:], torch.full((k,), N, device=device, dtype=torch.long)], dim=0)

            # renoise for the next step; fresh tail positions (uniform logits) renoise to Uniform(V)
            xt_tokens = pipeline.sample_logits_to_tokens(xt_logits)[None]  # (1, L)

        gen_tokens = pipeline.strip_thinking_tokens(torch.cat(committed))
        gen_texts = pipeline.processor.decode(gen_tokens, skip_special_tokens=True)
        
        return gen_texts


if __name__ == "__main__":
    set_seed(0)
    cli_evaluate()
