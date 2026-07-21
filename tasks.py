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

# git clone https://github.com/algorithmicsuperintelligence/openevolve.git
class CirclePacking:

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
        self.ref_code, self.best_reward = self.initial_code(), -math.inf
        self.data = [None]
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
        """Return the task prompt: embeds the best code so far as the code to rewrite."""
        return self.TASK_PROMPT.format(ref_code=self.ref_code)

    def evaluate(self, _, response: str) -> float:
        """Reward `response`: pull the rewritten code out of it and score that code with the openevolve
        example evaluator. Also ratchets the best code so far."""
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
        """Ratchet ref_code / best_reward against every process's newest (code, reward)."""
        for gathered_code, gathered_reward in gather_object([(code, reward)]):
            if gathered_reward > self.best_reward:
                self.ref_code, self.best_reward = gathered_code, gathered_reward


class Short:

    SYSTEM_PROMPT = "You are a helpful assistant."
    data = ["tell a story"]

    def prompt(self, data_id) -> str:
        return self.data[data_id]

    def evaluate(self, _, response: str) -> float:
        return 1-float(len(response))


class GSM8K:

    SYSTEM_PROMPT = inspect.cleandoc("""
        You are a helpful assistant. Think and response the final answer, enclose the final answer by <answer> </answer> tags.
    """)

    def __init__(self):
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        self.data = [{'question':x, 'answer':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]

    def prompt(self, data_id) -> str:
        """Return the user question for the item at data_id."""
        return self.data[data_id]["question"]

    def evaluate(self, data_id, response: str) -> float:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match is None: return 0
        answer = parse(match.group(1), extraction_config=[ExprExtractionConfig()])
        ground_truth = parse(self.data[data_id]["answer"], extraction_config=[ExprExtractionConfig()])
        return 1 if verify(answer, ground_truth) else 0


TASKS_CLS = {
    "circle-packing": CirclePacking,
    "short": Short,
    "gsm8k": GSM8K,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()
