import inspect
import re
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

TASKS_CLS = {
    "sudoku": Sudoku,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()