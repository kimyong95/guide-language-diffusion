import inspect
import importlib.util
import io
import contextlib
import math
import os
from functools import lru_cache
from pathlib import Path
import re
import tempfile
import torch
from accelerate.utils import gather_object
from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig

class CirclePacking:
    """OpenEvolve circle-packing task (n=26): the model evolves constructor code that places 26
    circles in the unit square to maximize the sum of radii. Seed code and evaluator are reused
    from the cloned openevolve repo / pip package. Every prompt asks the model to rewrite the best
    code found so far, which evaluate ratchets."""

    EXAMPLE = Path(__file__).resolve().parent / "openevolve" / "examples" / "circle_packing"

    SYSTEM_PROMPT = inspect.cleandoc("""
        You are an expert mathematician specializing in circle packing problems and computational
        geometry. Your task is to improve a constructor function that directly produces a specific
        arrangement of 26 circles in a unit square, maximizing the sum of their radii. The AlphaEvolve
        paper achieved a sum of 2.635 for n=26.

        Key geometric insights:
        - Circle packings often follow hexagonal patterns in the densest regions
        - Maximum density for infinite circle packing is pi/(2*sqrt(3)) ~ 0.9069
        - Edge effects make square container packing harder than infinite packing
        - Circles can be placed in layers or shells when confined to a square
        - Similar radius circles often form regular patterns, while varied radii allow better space use
        - Perfect symmetry may not yield the optimal packing due to edge effects

        Focus on designing an explicit constructor that places each circle in a specific position,
        rather than an iterative search algorithm.
    """)

    TASK_PROMPT = inspect.cleandoc("""
        # Task
        Rewrite the following code to maximize the sum of the 26 circle radii (higher score is better).

        ```python
        {ref_code}
        ```

        Rewrite the whole code, includes the main. Make sure your rewritten code maintains the same inputs and outputs as the original code, but with improved internal implementation, wraps in the python code block.
    """)

    def __init__(self):
        # Best code so far -- what every prompt asks the model to rewrite. Seeded with the openevolve
        # initial program, scored through evaluate like any response: raw code carries no ``` fence,
        # so the extraction below passes it straight to the evaluator.
        self.ref_code, self.best_reward = self.initial_code(), -math.inf
        self.data = [None]  # one dynamic prompt (rewrite the ratcheted best code); the item is unused
        self.evaluate(None, self.ref_code)

    @staticmethod
    @lru_cache(maxsize=1)
    def example_evaluator():
        spec = importlib.util.spec_from_file_location("circle_packing_evaluator", CirclePacking.EXAMPLE / "evaluator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    @lru_cache(maxsize=1)
    def initial_code() -> str:
        code = (CirclePacking.EXAMPLE / "initial_program.py").read_text(encoding="utf-8")
        return code

    def prompt(self, _) -> str:
        """Return the task prompt: embeds the best code so far as the code to rewrite (the system prompt
        is read separately from SYSTEM_PROMPT)."""
        return self.TASK_PROMPT.format(ref_code=self.ref_code)

    def evaluate(self, _, response: str) -> float:
        """Reward `response`: pull the rewritten code out of it and score that code with the openevolve
        example evaluator (runs it in a subprocess with a timeout and validates the packing; the
        combined score is sum_radii / 2.635 when valid). Also ratchets the best code so far."""
        from openevolve.utils.code_utils import parse_full_rewrite
        code = parse_full_rewrite(response, "python")

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            metrics = self.example_evaluator().evaluate(path)
        finally:
            os.unlink(path)
        reward = float(metrics.get("combined_score", 0.0))

        self.update_best(code, reward)
        return reward

    def update_best(self, code: str, reward: float) -> None:
        """Ratchet ref_code / best_reward against every process's newest (code, reward). Each rank
        calls evaluate the same number of times per epoch, so this gather is balanced; ranks scan the
        gathered pairs in the same order, so they all settle on the same best."""
        for gathered_code, gathered_reward in gather_object([(code, reward)]):
            if gathered_reward > self.best_reward:
                self.ref_code, self.best_reward = gathered_code, gathered_reward


class Short:
    """Quick smoke-test task: prompt the model to "tell a story" and reward the response length (number
    of characters). Nothing to extract or ratchet -- evaluate scores the raw generated text."""

    SYSTEM_PROMPT = "You are a helpful assistant."
    data = ["tell a story"]  # one fixed prompt -> a single item; GRPO must run with m=1 (one group)

    def prompt(self, data_id) -> str:
        return self.data[data_id]

    def evaluate(self, _, response: str) -> float:
        return 1-float(len(response))


class GSM8K:
    """GSM-Hard math-reasoning task for GRPO (kept under the GSM8K name / "gsm8k" key). A dataset of
    (Q, A) items with numeric answers -- the hard-numbers variant of GSM8K. The trainer draws grouped
    prompts from `self.data`, keyed by data id; `evaluate(data_id, response)` pulls the number inside the
    response's <answer> </answer> tags and scores +1/-1 by numeric match against the ground truth."""

    

    SYSTEM_PROMPT = inspect.cleandoc("""
        You are a helpful assistant. Think and response the final answer, enclose the final answer by <answer> </answer> tags.
    """)

    def __init__(self):
        # GSM-Hard: `input` is the question, `target` is the numeric ground-truth answer (a float).
        dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        self.data = [{"Q": q, "A": str(t)} for q, t in zip(dataset["input"], dataset["target"])]

    def prompt(self, data_id) -> str:
        """Return the user question for the (Q, A) item at data_id (the system prompt is read separately
        from SYSTEM_PROMPT)."""
        return self.data[data_id]["Q"]

    def evaluate(self, data_id, response: str) -> float:
        # Score the number inside the <answer> </answer> tags against the ground truth.
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match is None: return -1.0
        nums = re.findall(r'\d+\.\d+|\d+/\d+|\d+', match.group(1))
        if len(nums) == 0: return -1.0
        answer = parse(nums[-1], extraction_config=[ExprExtractionConfig()])
        ground_truth = parse(self.data[data_id]["A"], extraction_config=[ExprExtractionConfig()])
        return 1 if verify(answer, ground_truth) else -1


# git clone https://github.com/algorithmicsuperintelligence/openevolve.git
TASKS_CLS = {
    "circle-packing": CirclePacking,
    "short": Short,
    "gsm8k": GSM8K,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()