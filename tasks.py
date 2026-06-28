import inspect
import importlib.util
import io
import contextlib
from functools import lru_cache
from pathlib import Path
import re
import tempfile
import torch
from datasets import load_dataset

class Sudoku:

    PROMPT_TEMPLATE = inspect.cleandoc("""
        Solve the following Sudoku puzzle, where 0 represents the empty cells to be filled:
        {puzzle}
        
        Response 9x9 grid with no spaces, rows separated by newlines.
    """)

    # a row is exactly 9 digits (not part of a longer digit run); a grid is 9 such rows
    ROW_RE = r"(?<![0-9])[0-9]{9}(?![0-9])"
    GRID_RE = re.compile(rf"{ROW_RE}(?:[ \t]*\r?\n[ \t]*{ROW_RE}){{8}}")

    def __init__(self, puzzle_id=0):
        ds = load_dataset("sapientinc/sudoku-extreme",split="test",converters={"question": str, "answer": str},)
        row = ds.sort("rating")[int(puzzle_id)]
        self.puzzle = self.to_grid(row["question"].replace(".", "0"))
        self.ground_truth = row["answer"]

    @staticmethod
    def to_grid(digits: str) -> str:
        return "\n".join(digits[i:i + 9] for i in range(0, 81, 9))

    def evaluate_one(self, response: str) -> float:
        matches = self.GRID_RE.findall(response)
        if not matches:
            return 0.0
        solution = "".join(char for char in matches[-1] if char.isdigit())
        N = len(self.ground_truth)
        if len(solution) != N:
            return 0.0
        return sum(1 for i in range(N) if solution[i] == self.ground_truth[i]) / N

    def evaluate(self, responses: list[str]) -> torch.Tensor:
        return torch.tensor([self.evaluate_one(r) for r in responses], dtype=torch.float32)

    def prompt(self):
        return self.PROMPT_TEMPLATE.format(puzzle=self.puzzle)


class FunctionMinimization:

    PROMPT_TEMPLATE = inspect.cleandoc("""
        Minimize the function:

        f(x, y) = sin(x) * cos(y) + sin(x * y) + (x^2 + y^2) / 20

        Write Python code defining a function named run_search().
        run_search() must return either (x, y) or (x, y, value), where value is f(x, y).
    """)

    CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

    @staticmethod
    @lru_cache(maxsize=1)
    def example_evaluator():
        evaluator_path = (
            Path(__file__).resolve().parent
            / "openevolve"
            / "examples"
            / "function_minimization"
            / "evaluator.py"
        )
        spec = importlib.util.spec_from_file_location("function_minimization_evaluator", evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load evaluator from {evaluator_path}")

        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        return evaluator

    @classmethod
    def extract_code(cls, response: str) -> str:
        matches = cls.CODE_FENCE_RE.findall(response)
        if matches:
            return matches[-1].strip()
        return response.strip()

    def evaluate_one(self, response: str) -> float:
        code = self.extract_code(response)
        if not code:
            return 0.0

        evaluator = self.example_evaluator()
        with tempfile.TemporaryDirectory(prefix="function_minimization_") as tmpdir:
            program_path = Path(tmpdir) / "candidate.py"
            program_path.write_text(code, encoding="utf-8")

            # The OpenEvolve evaluator is verbose by design; keep task rewards quiet.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = evaluator.evaluate(str(program_path))

        return float(result.metrics.get("combined_score", 0.0))

    def evaluate(self, responses: list[str]) -> torch.Tensor:
        return torch.tensor([self.evaluate_one(r) for r in responses], dtype=torch.float32)

    def prompt(self):
        return self.PROMPT_TEMPLATE

# git clone https://github.com/algorithmicsuperintelligence/openevolve.git
TASKS_CLS = {
    "sudoku": Sudoku,
    "func-min": FunctionMinimization,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()