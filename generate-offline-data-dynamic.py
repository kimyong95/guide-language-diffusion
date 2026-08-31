"""    accelerate launch --num_processes 4 --main_process_port 0 generate-offline-data-dynamic.py"""

import argparse
import os
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm

from pipeline import Pipeline
from tasks import get_reward_fn
from utils import gather

MODEL = "Qwen/Qwen3-8B"
K_INIT = 1           # a round smaller than the process count leaves the tail of the processes idle, so K_INIT = G wastes nothing
MAX_NEW_TOKENS = 16384
TEMPERATURE = 0.6    # Qwen3's own thinking-mode settings, with the two below; its MinP=0 filters nothing, so no min_p is set
TOP_P = 0.95
TOP_K = 20
ENABLE_THINKING = True
SEED = 0

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="aime-2024")
parser.add_argument("--k-max", type=int, default=4096)
args = parser.parse_args()
TASK, K_MAX = args.task, args.k_max

OUT_PATH = f"test-data/{TASK}-pass@{K_MAX*2}.pt"   # the doubling redraws from scratch each round, so a question costs up to 2 * K_MAX samples

accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))])   # a round is one full generation long, and the processes only meet at its end
set_seed(SEED, device_specific=True)
G, g = accelerator.num_processes, accelerator.process_index

task = get_reward_fn(TASK)
pipeline = Pipeline(MODEL, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
if accelerator.is_main_process:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

data = {"data_ids": [], "prompt_tokens": [], "generated_tokens": []}   # (N,) and two 2D ragged lists (N, Lp) / (N, Lg), one entry per solved question
progress = tqdm(range(len(task.data)), desc="Sampling", disable=not accelerator.is_main_process)
for data_id in progress:
    prompt_tokens, = pipeline.texts_to_tokens([task.prompt(data_id)], system_prompt=task.SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING)

    k, correct = K_INIT, None
    while correct is None and k <= K_MAX:
        k_local = k // G + (g < k % G)   # the round's k copies split over the processes, as evenly as k allows
        correct_local = None
        if k_local:
            generated_output = pipeline.generate([prompt_tokens] * k_local, max_new_tokens=MAX_NEW_TOKENS)   # k_local copies of one prompt, one completion each
            correct_local = next((tokens for tokens, text in zip(generated_output.tokens, generated_output.texts) if task.evaluate(data_id, text)), None)
        correct = next((tokens for tokens in gather([correct_local]) if tokens is not None), None)   # every process leaves the round holding the same answer, so they stay in step
        k *= 2

    progress.set_postfix(kept=len(data["data_ids"]), k=k // 2)
    if correct is None:   # unsolved at K_MAX: the question contributes no entry
        continue

    data["data_ids"].append(data_id)
    data["prompt_tokens"].append(prompt_tokens)
    data["generated_tokens"].append(correct)

    if accelerator.is_main_process:
        torch.save({**data, "data_ids": torch.tensor(data["data_ids"])}, OUT_PATH + ".tmp")   # write-then-rename: a crash mid-save leaves the last good file
        os.replace(OUT_PATH + ".tmp", OUT_PATH)

accelerator.print(f"{len(data['data_ids'])}/{len(task.data)} questions solved -> {OUT_PATH}")
