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


GENERAL_PROMPT_TEMPLATE = inspect.cleandoc("""
    # Reference Program
    ```{language}
    {program}
    ```

    # Task
    Rewrite the program to {instruction}.
    Provide the complete new program code.

    IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.

    Response in the format:
    ```{language}
    # Your rewritten program here
    ```
""")


class FunctionMinimization:

    LANGUAGE = "python"
    INSTRUCTION = "minimize the function f(x, y) = sin(x) * cos(y) + sin(x * y) + (x^2 + y^2) / 20"

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

    @staticmethod
    @lru_cache(maxsize=1)
    def initial_program() -> str:
        program_path = (
            Path(__file__).resolve().parent
            / "openevolve"
            / "examples"
            / "function_minimization"
            / "initial_program.py"
        )
        code = program_path.read_text(encoding="utf-8")

        # Strip openevolve-specific marker/annotation comments (whole line each).
        markers = (
            "# EVOLVE-BLOCK-START",
            "# EVOLVE-BLOCK-END",
            "# This part remains fixed (not evolved)",
            "# AlphaEvolve improved this to",
        )
        lines = [l for l in code.splitlines() if not any(m in l for m in markers)]
        return "\n".join(lines)

    @classmethod
    def extract_program(cls, response: str) -> str | None:
        from openevolve.utils.code_utils import parse_full_rewrite

        # Rewrite mode: pull the complete rewritten program out of the response.
        code = parse_full_rewrite(response, cls.LANGUAGE)
        return code or None

    def evaluate_one(self, response: str) -> float:
        code = self.extract_program(response)
        if code is None:
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
        return GENERAL_PROMPT_TEMPLATE.format(
            language=self.LANGUAGE,
            program=self.initial_program(),
            instruction=self.INSTRUCTION,
        )

# git clone https://github.com/algorithmicsuperintelligence/openevolve.git
TASKS_CLS = {
    "sudoku": Sudoku,
    "func-min": FunctionMinimization,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()