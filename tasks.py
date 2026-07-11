import inspect
import importlib.util
import io
import contextlib
import os
from functools import lru_cache
from pathlib import Path
import re
import tempfile
import torch
from datasets import load_dataset

class CirclePacking:
    """OpenEvolve circle-packing task (n=26): the model evolves constructor code that places 26
    circles in the unit square to maximize the sum of radii. Seed code and evaluator are reused
    from the cloned openevolve repo / pip package. This base holds everything shared by the rewrite
    and edit variants; subclasses supply only the task instruction (TASK_MESSAGE) and how a response
    turns into the next code (extract_code)."""

    EXAMPLE = Path(__file__).resolve().parent / "openevolve" / "examples" / "circle_packing"

    SYSTEM_MESSAGE = inspect.cleandoc("""
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

    TASK_MESSAGE = inspect.cleandoc("""
        # Task
        Rewrite the following code to maximize the sum of the 26 circle radii (higher score is better).

        ```python
        {ref_code}
        ```
                                    
        Rewrite the whole code, includes the main. Make sure your rewritten code maintains the same inputs and outputs as the original code, but with improved internal implementation, wraps in the python code block.
    """)

    def __init__(self):
        self.ref_code = self.initial_code()  # reference code to rewrite; ratcheted by the caller

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

    def prompt(self) -> tuple[str, str]:
        """Return (system message, task message). The task message embeds the current ref_code as
        the code to rewrite."""
        return self.SYSTEM_MESSAGE, self.TASK_MESSAGE.format(ref_code=self.ref_code)

    @staticmethod
    def extract_code(response: str) -> str:
        from openevolve.utils.code_utils import parse_full_rewrite
        return parse_full_rewrite(response, "python")

    def evaluate_code(self, code: str) -> float:
        """Reward the code via the openevolve example evaluator (runs it in a subprocess with a
        timeout and validates the packing); the combined score is sum_radii / 2.635 when valid."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            metrics = self.example_evaluator().evaluate(path)
        finally:
            os.unlink(path)
        return float(metrics.get("combined_score", 0.0))


class Dog:
    """Quick smoke-test task: prompt the model to "tell a story" and reward the number of times the
    word "dog" appears in the response. No reference code (ref_code is None) and extract_code is the
    identity, so the raw generated text flows straight into evaluate_code."""

    SYSTEM_MESSAGE = "You are a helpful assistant."
    TASK_MESSAGE = "tell a story"

    def __init__(self):
        self.ref_code = None  # no reference code; kept for interface compatibility

    @staticmethod
    def initial_code() -> str:
        return ""  # empty seed -> evaluate_code("") == 0.0

    def prompt(self) -> tuple[str, str]:
        return self.SYSTEM_MESSAGE, self.TASK_MESSAGE

    @staticmethod
    def extract_code(response: str) -> str:
        return response  # identity: reward the raw generated text

    def evaluate_code(self, code: str) -> float:
        return float(code.count("dog"))


# git clone https://github.com/algorithmicsuperintelligence/openevolve.git
TASKS_CLS = {
    "circle-packing": CirclePacking,
    "dog": Dog,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()