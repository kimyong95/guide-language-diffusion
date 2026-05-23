import inspect
from itertools import islice
import csv
import re
import torch

class Sudoku:

    QUESTION_PROMPT_TEMPLATE = inspect.cleandoc("""
        Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

        **Rules:**
        - Fill empty cells with digits 1-4.
        - Each row must contain digits 1-4 exactly once.
        - Each column must contain digits 1-4 exactly once.
        - Each 2x2 box must contain digits 1-4 exactly once.

        **Example:**
        Puzzle: 0401002010030310
        This puzzle grid looks like this:
        0 4 | 0 1
        0 0 | 2 0
        ----+----
        1 0 | 0 3
        0 3 | 1 0

        Solution: 2431312412434312
        The solved grid looks like this:
        2 4 | 3 1
        3 1 | 2 4
        ----+----
        1 2 | 4 3
        4 3 | 1 2

        **Important:** Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

        Respond in this exact format:
        <reasoning>
        Your step-by-step solving process
        </reasoning>
        <answer>
        [16-character solution string with no spaces or separators]
        </answer>                                

        Now, solve the following Sudoku puzzle: {puzzle}
    """)

    def __init__(self, puzzle_id):
        with open('data/4x4_sudoku_unique_puzzles.csv', 'r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file)
            self.puzzle, self.ground_truth = next(islice(reader, int(puzzle_id)+1, int(puzzle_id)+2), None)
        
    def evaluate_one(self, response: str) -> float:
        matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not matches:
            return 0.0
        solution = "".join(char for char in matches[-1].strip() if char.isdigit())
        N = len(self.ground_truth)
        if len(solution) != N:
            return 0.0
        return sum(1 for i in range(N) if solution[i] == self.ground_truth[i]) / N

    def evaluate(self, responses: list[str]) -> torch.Tensor:
        return torch.tensor([self.evaluate_one(r) for r in responses], dtype=torch.float32)

    def question_prompt(self):
        return self.QUESTION_PROMPT_TEMPLATE.format(puzzle=self.puzzle)

TASKS_CLS = {
    "sudoku": Sudoku,
}

def get_reward_fn(key: str):
    name, _, arg = key.partition(":")
    cls = TASKS_CLS[name]
    return cls(arg) if arg else cls()