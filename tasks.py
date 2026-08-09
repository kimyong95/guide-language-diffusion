import inspect
import re
from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

import problems


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


class CirclePacking:
    """problems.CirclePacking behind the task interface: one prompt, so the dataset is a single dummy item."""

    SYSTEM_PROMPT = problems.CirclePacking.SYSTEM_PROMPT

    def __init__(self):
        self.problem = problems.CirclePacking()
        self.data = [None]

    def prompt(self, data_id: int) -> str:
        return self.problem.prompt()

    def evaluate(self, data_id: int, response: str) -> float:
        return self.problem.evaluate(response)


TASKS_CLS = {
    "gsm8k": GSM8K,
    "aime-2024": AIME2024,
    "math-500": MATH500,
    "circle-packing": CirclePacking,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()
