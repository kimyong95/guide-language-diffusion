import inspect
import json
import re
from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig


class MathTask:
    """The Qwen3 model card's math prompt and math_verify grading of the whole response, \\boxed{} first; subclasses only supply self.data.

    No system prompt: verl feeds the parquet's lone user turn straight to apply_chat_template,
    leaving whatever system turn the model's own template injects (none, for Qwen3).
    """

    SYSTEM_PROMPT = None

    PROMPT_TEMPLATE = inspect.cleandoc("""
        {question}
        Please reason step by step, and put your final answer within \\boxed{{}}.
    """)

    EXTRACTION_CONFIG = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]

    def prompt(self, data_id: int) -> str:
        return self.PROMPT_TEMPLATE.format(question=self.data[data_id]["question"])

    def evaluate(self, data_id: int, response: str) -> float:
        answer = parse(response, extraction_config=self.EXTRACTION_CONFIG)
        ground_truth = parse(f"${self.data[data_id]['answer']}$", extraction_config=self.EXTRACTION_CONFIG)
        return 1 if verify(ground_truth, answer) else 0


class DAPOMath17K(MathTask):

    def __init__(self):
        dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "all", split="train")
        self.data = [{'question':x, 'answer':y.strip()} for x,y in zip(dataset['prompt'], dataset['solution'])]


class AIMOAIME(MathTask):
    """One year of AIME I and II out of AIMO's validation set, 30 problems; subclasses only set YEAR.

    The set holds 2022 through 2024 and names the year nowhere but the AoPS url each problem was
    taken from, so that is what the year is read off.
    """

    YEAR = None

    def __init__(self):
        dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
        self.data = [{'question':x, 'answer':y.strip()} for x,y,z in zip(dataset['problem'], dataset['answer'], dataset['url']) if f"/{self.YEAR}_AIME" in z]


class AIME2022(AIMOAIME):

    YEAR = 2022


class AIME2023(AIMOAIME):

    YEAR = 2023


class AIME2024(AIMOAIME):

    YEAR = 2024


class AIME2025(MathTask):

    def __init__(self):
        dataset = load_dataset("MathArena/aime_2025", split="train")
        self.data = [{'question':x, 'answer':str(y)} for x,y in zip(dataset['problem'], dataset['answer'])]


class AIME2026(MathTask):

    def __init__(self):
        dataset = load_dataset("MathArena/aime_2026", split="train")
        self.data = [{'question':x, 'answer':str(y)} for x,y in zip(dataset['problem'], dataset['answer'])]


class MATH500(MathTask):

    def __init__(self):
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        self.data = [{'question':x, 'answer':y.strip()} for x,y in zip(dataset['problem'], dataset['answer'])]


TASKS_CLS = {
    "aime-2022": AIME2022,
    "aime-2023": AIME2023,
    "aime-2024": AIME2024,
    "aime-2025": AIME2025,
    "aime-2026": AIME2026,
    "math-500": MATH500,
    "dapo-math-17k": DAPOMath17K,
}

SLICE_STR_PATTERN = r"([^\[]+)(?:\[([-\d:]+)\])?"

def slice_data(data, slice_str: str):
    """
    Args:
        data: list
        index: str, what stands inside the key's square bracket, e.g. ":10", "10:20", "0"
    """
    bounds = [int(b) if b else None for b in slice_str.split(":")]
    return data[slice(*bounds)] if len(bounds) > 1 else [data[bounds[0]]]

def get_reward_fn(task_name: str):
    """
    Args:
        key: str, a TASKS_CLS name with an optional Python subscript, e.g. "math-500[:10]"
    """
    name, slide_str = re.fullmatch(SLICE_STR_PATTERN, task_name).groups()
    task = TASKS_CLS[name]()
    if slide_str is not None:
        task.data = slice_data(task.data, slide_str)
    return task
