import inspect
import re
from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

import problems


class MathTask:
    """DAPO's prompt format and Answer:-line grading; subclasses only supply self.data.

    No system prompt: verl feeds the parquet's lone user turn straight to apply_chat_template,
    leaving whatever system turn the model's own template injects (none, for Qwen3).
    """

    SYSTEM_PROMPT = None

    PROMPT_TEMPLATE = inspect.cleandoc("""
        Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

        {question}

        Remember to put your answer on its own line after "Answer:".
    """)

    ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

    EXTRACTION_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]

    def prompt(self, data_id: int) -> str:
        return self.PROMPT_TEMPLATE.format(question=self.data[data_id]["question"])

    def evaluate(self, data_id: int, response: str) -> float:
        matches = re.findall(self.ANSWER_PATTERN, response)
        if not matches: return 0
        answer = parse(f"${matches[-1].strip()}$", extraction_config=self.EXTRACTION_CONFIG)
        ground_truth = parse(f"${self.data[data_id]['answer']}$", extraction_config=self.EXTRACTION_CONFIG)
        return 1 if verify(ground_truth, answer) else 0


class DAPOMath17K(MathTask):

    def __init__(self):
        dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "all", split="train")
        self.data = [{'question':x, 'answer':y.strip()} for x,y in zip(dataset['prompt'], dataset['solution'])]


class AIME2024(MathTask):
    """verl's DAPO validation file: 30 problems x 32 copies, deduplicated since repetition is the sampler's job."""

    def __init__(self):
        dataset = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")
        data = {x['index']: {'question':x['raw_problem'], 'answer':y['ground_truth'].strip()} for x,y in zip(dataset['extra_info'], dataset['reward_model'])}
        self.data = list(data.values())


class MATH500(MathTask):

    def __init__(self):
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        self.data = [{'question':x, 'answer':y.strip()} for x,y in zip(dataset['problem'], dataset['answer'])]


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
    "aime-2024": AIME2024,
    "math-500": MATH500,
    "dapo-math-17k": DAPOMath17K,
    "circle-packing": CirclePacking,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()
