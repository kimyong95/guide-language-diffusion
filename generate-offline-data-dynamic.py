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
MAX_NEW_TOKENS = 16384
TEMPERATURE = 0.6    # Qwen3's own thinking-mode settings, with the two below; its MinP=0 filters nothing, so no min_p is set
TOP_P = 0.95
TOP_K = 20
ENABLE_THINKING = True
SEED = 0

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="aime-2024")
parser.add_argument("--k", type=int, default=16)
args = parser.parse_args()
TASK, K = args.task, args.k

OUT_PATH = f"test-data/{TASK}-pass@{K}.pt"   # a question gets one draw per step and at most K steps, so solving it at all is exactly pass@K

accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))])   # a step is one full generation long, and the processes only meet at its end
set_seed(SEED, device_specific=True)
G, g = accelerator.num_processes, accelerator.process_index

task = get_reward_fn(TASK)
pipeline = Pipeline(MODEL, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
if accelerator.is_main_process:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

data = {"data_ids": [], "prompt_tokens": [], "generated_tokens": []}   # (N,) and two 2D ragged lists (N, Lp) / (N, Lg), one entry per solved question
unsolved = list(range(len(task.data)))
progress = tqdm(range(1, K + 1), desc="Sampling", disable=not accelerator.is_main_process)
for step in progress:
    local = unsolved[g * len(unsolved) // G:(g + 1) * len(unsolved) // G]   # a contiguous block of what is still open, empty once fewer questions remain than there are processes

    solved = []
    if local:
        generated_output = pipeline.generate(pipeline.texts_to_tokens([task.prompt(data_id) for data_id in local], system_prompt=task.SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING), max_new_tokens=MAX_NEW_TOKENS)   # one draw for each question in the block
        solved = [(data_id, tokens) for data_id, tokens, text in zip(local, generated_output.tokens, generated_output.texts) if task.evaluate(data_id, text)]

    solved = gather(solved)   # every process leaves the step holding the same answers, so they stay in step
    data["data_ids"] += [data_id for data_id, _ in solved]
    data["prompt_tokens"] += pipeline.texts_to_tokens([task.prompt(data_id) for data_id, _ in solved], system_prompt=task.SYSTEM_PROMPT, enable_thinking=ENABLE_THINKING)
    data["generated_tokens"] += [tokens for _, tokens in solved]
    solved_ids = {data_id for data_id, _ in solved}
    unsolved = [data_id for data_id in unsolved if data_id not in solved_ids]

    progress.set_postfix(solved=len(data["data_ids"]), unsolved=len(unsolved))
    if accelerator.is_main_process:
        torch.save({**data, "data_ids": torch.tensor(data["data_ids"])}, OUT_PATH + ".tmp")   # write-then-rename: a crash mid-save leaves the last good file
        os.replace(OUT_PATH + ".tmp", OUT_PATH)

    if not unsolved:
        break

accelerator.print(f"{len(data['data_ids'])}/{len(task.data)} questions solved in {step} steps -> {OUT_PATH}")
