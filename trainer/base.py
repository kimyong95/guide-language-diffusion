import math
import os
import sys
import wandb
from pipeline_vllm import Pipeline
import tasks



class BaseTrainer:

    def __init__(self, config):
        self.config = config
        self.setup_wandb()
        self.setup_task()
        self.setup_pipeline()
        self.log_code()
        self.text_table = wandb.Table(
            columns=["objective-evaluations", "idx", "reward", "text"],
            log_mode="INCREMENTAL",
        )
        self.best_reward = -math.inf

    def setup_wandb(self):
        wandb.init(project="guide-language-diffusion", name=self.config.run_name, config=self.config.to_dict())

    def setup_task(self):
        self.task = tasks.get_reward_fn(self.config.task)

    def setup_pipeline(self, **kwargs):
        """Builds config.model's vLLM pipeline; kwargs go to vllm.LLM."""
        self.pipeline = Pipeline(self.config.model.name, self.config.model.max_total_tokens, seed=self.config.seed, **kwargs)

    def log_code(self):
        cwd = os.path.abspath(os.getcwd())
        imported_py_files = set()
        for module in sys.modules.values():
            path = getattr(module, "__file__", None)
            if path and path.endswith(".py"):
                abs_path = os.path.abspath(path)
                if abs_path.startswith(cwd):
                    imported_py_files.add(abs_path)

        wandb.run.log_code(".", include_fn=lambda path: path in imported_py_files)

    def log_rewards(self, objective_evaluations, rewards, extra={}):
        self.best_reward = max(self.best_reward, rewards.max().item())
        log_dict = {
            "objective-evaluations": objective_evaluations,
            "rewards": rewards.mean().item(),
            "rewards-best": rewards.max().item(),
            "best-so-far": self.best_reward,
            **extra,
        }
        wandb.log(log_dict)

    def log_texts(self, objective_evaluations, rewards, texts, extra={}):
        for idx, (text, reward) in enumerate(zip(texts, rewards)):
            self.text_table.add_data(objective_evaluations, idx, reward.item(), text)
        log_dict = {
            "objective-evaluations": objective_evaluations,
            "texts": self.text_table,
            **extra,
        }
        wandb.log(log_dict)
