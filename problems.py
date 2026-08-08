import importlib.util
import inspect
import os
from functools import lru_cache
from pathlib import Path
import tempfile
import yaml

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

    def prompt(self, ref_data = None) -> str:

        if ref_data is not None:
            ref_code = parse_full_rewrite(ref_data, "python")
        else:
            ref_code = self.ref_code

        return self.TASK_PROMPT.format(ref_code=ref_code, language="python")

    def evaluate(self, response: str) -> float:

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


PROBLEMS_CLS = {
    "circle-packing": CirclePacking,
}

def get_problem(key: str):
    name, _, arg = key.partition(":")
    cls = PROBLEMS_CLS[name]
    return cls(arg) if arg else cls()
