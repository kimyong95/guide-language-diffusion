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
import yaml
from accelerate.utils import gather_object
from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

# git clone https://github.com/algorithmicsuperintelligence/openevolve.git
from openevolve.utils.code_utils import parse_full_rewrite
class CirclePacking:

    EXAMPLE = Path(__file__).resolve().parent / "openevolve" / "examples" / "circle_packing"
    CONFIG = EXAMPLE / "config_phase_1.yaml"

    SYSTEM_PROMPT = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["prompt"]["system_message"].strip()

    # from openevolve/openevolve/prompts/defaults/full_rewrite_user.txt
    TASK_PROMPT = inspect.cleandoc("""
        # Current Program
        ```{language}
        {ref_code}
        ```

        # Task
        Rewrite the program to improve its sum_radii.
        Provide the complete new program code.

        IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
        as the original program, but with improved internal implementation.

        ```{language}
        # Your rewritten program here
        ```
    """)

    def __init__(self):
        self.ref_code = self.initial_code()
        self.data = [self.TASK_PROMPT]

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

    def prompt(self, data_id: int, ref_data = None) -> str:

        if ref_data is not None:
            ref_code = parse_full_rewrite(ref_data, "python")
        else:
            ref_code = self.ref_code

        return self.data[data_id].format(ref_code=ref_code, language="python")

    def evaluate(self, data_id: int, response: str) -> float:
        
        code = parse_full_rewrite(response, "python")

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            metrics = self.example_evaluator().evaluate(path)
        finally:
            os.unlink(path)
        reward = float(metrics.get("combined_score", 0.0))

        return reward


class GSM8K:

    SYSTEM_PROMPT = inspect.cleandoc("""
        You are a helpful assistant. Think and response the final answer, enclose the final answer by <answer> </answer> tags.
    """)

    EXTRACTION_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]

    def __init__(self):
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        self.data = [{'question':x, 'answer':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]

    def prompt(self, data_id: int) -> str:
        """Return the user question for the item at data_id."""
        return self.data[data_id]["question"]

    def evaluate(self, data_id: int, response: str) -> float:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match is None: return 0
        answer = parse(f"${match.group(1)}$", extraction_config=self.EXTRACTION_CONFIG)
        ground_truth = parse(f"${self.data[data_id]['answer']}$", extraction_config=self.EXTRACTION_CONFIG)
        return 1 if verify(answer, ground_truth) else 0


class AIME2024:

    SYSTEM_PROMPT = inspect.cleandoc("""
        You are a helpful assistant. Think and response the final answer, enclose the final answer by <answer> </answer> tags.
    """)

    EXTRACTION_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]

    def __init__(self):
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        self.data = [{'question':x, 'answer':y.strip()} for x,y in zip(dataset['problem'], dataset['answer'])]

    def prompt(self, data_id: int) -> str:
        """Return the user question for the item at data_id."""
        return self.data[data_id]["question"]

    def evaluate(self, data_id: int, response: str) -> float:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match is None: return 0
        answer = parse(f"${match.group(1)}$", extraction_config=self.EXTRACTION_CONFIG)
        ground_truth = parse(f"${self.data[data_id]['answer']}$", extraction_config=self.EXTRACTION_CONFIG)
        return 1 if verify(answer, ground_truth) else 0


class MATH500:

    SYSTEM_PROMPT = inspect.cleandoc("""
        You are a helpful assistant. Think and response the final answer, enclose the final answer by <answer> </answer> tags.
    """)

    EXTRACTION_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]

    def __init__(self):
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        self.data = [{'question':x, 'answer':y.strip()} for x,y in zip(dataset['problem'], dataset['answer'])]

    def prompt(self, data_id: int) -> str:
        """Return the user question for the item at data_id."""
        return self.data[data_id]["question"]

    def evaluate(self, data_id: int, response: str) -> float:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match is None: return 0
        answer = parse(f"${match.group(1)}$", extraction_config=self.EXTRACTION_CONFIG)
        ground_truth = parse(f"${self.data[data_id]['answer']}$", extraction_config=self.EXTRACTION_CONFIG)
        return 1 if verify(answer, ground_truth) else 0


TASKS_CLS = {
    "circle-packing": CirclePacking,
    "gsm8k": GSM8K,
    "aime-2024": AIME2024,
    "math-500": MATH500,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()
