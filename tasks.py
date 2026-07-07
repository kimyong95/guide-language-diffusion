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
    """OpenEvolve circle-packing task (n=26): the model evolves a constructor program that places 26
    circles in the unit square to maximize the sum of radii. Seed program and evaluator are reused
    from the cloned openevolve repo / pip package. This base holds everything shared by the rewrite
    and edit variants; subclasses supply only the task instruction (TASK_MESSAGE) and how a response
    turns into the next program (extract_program)."""

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
        Rewrite the program to maximize the sum of the 26 circle radii (higher score is better).
        Provide the complete new program code.

        IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
                                    
        ```python
        # Your rewritten program here
        ```
    """)

    @staticmethod
    @lru_cache(maxsize=1)
    def example_evaluator():
        spec = importlib.util.spec_from_file_location("circle_packing_evaluator", CirclePacking.EXAMPLE / "evaluator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    @lru_cache(maxsize=1)
    def initial_program() -> str:
        code = (CirclePacking.EXAMPLE / "initial_program.py").read_text(encoding="utf-8")
        return code

    def build_prompt(self, programs: list = None) -> str:
        """Prompt from a list of (code, reward): programs[0] is the current program, the rest are
        prior programs for reference. Renders only the reward score (never other metric floats), so
        the prompt is a deterministic function of the archive."""
        
        if programs is None:
            programs = [(self.initial_program(), 0.0)]

        sections = [self.SYSTEM_MESSAGE]
        for idx, (code, reward) in enumerate(programs):
            header = "Current program" if idx == 0 else f"Prior program [{idx}]"
            sections.append(f"# {header} (score={reward:.4f})\n```python\n{code}\n```")
        sections.append(self.TASK_MESSAGE)
        return "\n\n".join(sections)

    @staticmethod
    def extract_program(response: str) -> str:
        from openevolve.utils.code_utils import parse_full_rewrite
        return parse_full_rewrite(response, "python")

    def evaluate_program(self, code: str) -> float:
        """Reward the program via the openevolve example evaluator (runs it in a subprocess with a
        timeout and validates the packing); the combined score is sum_radii / 2.635 when valid."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            metrics = self.example_evaluator().evaluate(path)
        finally:
            os.unlink(path)
        return float(metrics.get("combined_score", 0.0))


# git clone https://github.com/algorithmicsuperintelligence/openevolve.git
TASKS_CLS = {
    "circle-packing": CirclePacking,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()