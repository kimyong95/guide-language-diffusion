'''
lm-evaluation-harness adapter for the DiffusionGemma block-diffusion model.

Implements `generate_until` only. Likelihood-based requests (`loglikelihood` /
`loglikelihood_rolling`) are not applicable to a diffusion model and raise
NotImplementedError, so this adapter works with generative tasks (e.g. gsm8k) that
grade on the produced answer string.

The generation recipe is lifted from test-parallel.py, with the tensor-parallel scaffolding
(device_mesh, tp_plan, dist.broadcast) stripped away. Data parallelism is supported: launch
with `accelerate launch --num_processes N eval-gemma.py ...` and each process holds a full
copy of the model on its own GPU while lm-eval shards the eval docs across ranks and gathers
the per-sample results back to rank 0 for aggregation.
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


@register_model("diffusiongemma")
class DiffusionGemmaEvalHarness(LM):
    def __init__(
        self,
        model_name="google/diffusiongemma-26B-A4B-it",
        gen_length=256,
        num_inference_steps=48,
        max_blocks=24,
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
            gen_length: decoder canvas length, i.e. tokens produced per block.
            num_inference_steps: diffusion denoising steps per block.
            max_blocks: upper bound on blocks; generation stops early on EOS.
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
        num_inference_steps = int(num_inference_steps)
        self.max_blocks = int(max_blocks)
        self.enable_thinking = enable_thinking

        self.pipeline = DiffusionGemmaPipeline(
            model_name,
            entropy_bound=float(entropy_bound),
            confidence_threshold=float(confidence_threshold),
            t_min=float(t_min),
            t_max=float(t_max),
            gen_length=gen_length,
        )
        self.pipeline.model.eval()

        # set_timesteps also records scheduler.num_inference_steps, which temperature() needs.
        self.timesteps = self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.pipeline.device)

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

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        pipeline = self.pipeline

        answer_list = []
        for req in tqdm(requests, desc="Generating..."):
            context = req.args[0]
            until = req.args[1].get("until", []) if len(req.args) > 1 else []

            prompt_tokens = pipeline.build_prompt_tokens(context, enable_thinking=self.enable_thinking)
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
            answer = pipeline.processor.decode(gen_tokens, skip_special_tokens=True)
            answer_list.append(answer)
        return answer_list


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
