import inspect
import json
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


class LiveBenchMath(MathTask):
    """LiveBench's math split, graded the way livebench/process_results/math grades it.

    Every row carries its own prompt, already naming the answer format its subtask is scored on,
    so prompt hands it over untouched and PROMPT_TEMPLATE goes unused. Two departures from
    LiveBench: AMPS_Hard's sympy comparison and the GPT tiebreaker behind it become math_verify,
    and olympiad is all-or-nothing where LiveBench scores the fraction of positions correct,
    since pass@k has nothing to do with a fractional reward.
    """

    CHOICE_PATTERN = r"\\textbf{\(([A-E])\)\s?}(.*?)(?:\\qquad|\$)"    # a letter and the value printed beside it in the question's own choice list

    def __init__(self):
        dataset = load_dataset("livebench/math", split="test")
        self.data = [{'question':x[0], 'answer':y.strip(), 'task':z} for x,y,z in zip(dataset['turns'], dataset['ground_truth'], dataset['task'])]

    def prompt(self, data_id: int) -> str:
        return self.data[data_id]["question"]

    def evaluate(self, data_id: int, response: str) -> float:
        question = self.data[data_id]
        return {"math_comp": self.evaluate_math_comp, "AMPS_Hard": self.evaluate_amps_hard, "olympiad": self.evaluate_olympiad}[question["task"]](question, response)

    def evaluate_math_comp(self, question: dict, response: str) -> float:
        ground_truth = question["answer"]
        if ground_truth.isdigit():
            return 1 if ground_truth in response[-50:] else 0    # AIME: the prompt asks for the three digits as the last thing written

        if ground_truth * 4 in response:    # the prompt asks for the letter five times over, and four in a row cannot be an accident
            return 1

        if self.last_boxed(response).replace("\\text{", "").replace("}", "").replace("\\", "").strip().lower() == ground_truth.lower():
            return 1

        value = dict(re.findall(self.CHOICE_PATTERN, question["question"])).get(ground_truth, "").strip().strip("$").strip("~")
        if value and value in response[-(20 + len(value)):]:    # the tail names the answer's value rather than its letter
            return 1

        last_line = response.strip().split("\n")[-1].strip().replace("*", "")
        parenthesized = re.search(r"\((.*?)\)", last_line)
        return 1 if last_line.lower() == ground_truth.lower() or (parenthesized and parenthesized.group(1).lower() == ground_truth.lower()) else 0

    def evaluate_amps_hard(self, question: dict, response: str) -> float:
        response = re.sub(r"\s*\+\s*[Cc]\b", "", response)    # the integral rows' ground truth carries no constant of integration
        answer = parse(response, extraction_config=self.EXTRACTION_CONFIG)
        ground_truth = parse(f"${question['answer']}$", extraction_config=self.EXTRACTION_CONFIG)
        return 1 if verify(ground_truth, answer) else 0

    def evaluate_olympiad(self, question: dict, response: str) -> float:
        return 1 if self.extract_expression_ids(response) == [int(n) for n in question["answer"].split(",")] else 0

    def last_boxed(self, response: str) -> str:
        """
        Args:
            response: str

        Returns:
            str, what the last \\boxed{} holds, matched by counting braces so a \\text{} nested
            inside it survives; "" when the response boxes nothing.
        """
        start = response.rfind("\\boxed{")
        depth = 0
        for i in range(start + len("\\boxed"), len(response) if start >= 0 else 0):
            depth += (response[i] == "{") - (response[i] == "}")
            if depth == 0:
                return response[start + len("\\boxed{"):i]
        return ""

    def extract_expression_ids(self, response: str):
        """
        Args:
            response: str

        Returns:
            list of int, the expression identifiers filling the masked slots, in the order given;
            empty when none of the places the answer is allowed to sit parses as a list of numbers.
        """
        candidates = [response.lower().split("answer:")[-1].strip().split("\n")[0]] if "answer:" in response.lower() else []    # stripped first, so an empty Answer: line falls through to the one below it
        candidates += [self.last_boxed(response), response.strip().split("\n")[-1]]
        for candidate in candidates:
            ids = [re.sub(r"\D", "", token) for token in candidate.split(",")]
            if ids and all(ids):
                return [int(i) for i in ids]
        return []


TASKS_CLS = {
    "aime-2022": AIME2022,
    "aime-2023": AIME2023,
    "aime-2024": AIME2024,
    "aime-2025": AIME2025,
    "aime-2026": AIME2026,
    "math-500": MATH500,
    "dapo-math-17k": DAPOMath17K,
    "livebench-math": LiveBenchMath,
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
