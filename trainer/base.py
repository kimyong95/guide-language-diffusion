import math
import os
import random
import sys
import numpy as np
import torch
import wandb
from pipeline import Pipeline
import tasks



class BaseTrainer:

    def __init__(self, config):
        self.config = config
        self.setup_wandb()
        self.setup_task()
        self.setup_pipeline()
        self.set_seed() # after setup_pipeline to prevent random port crash across same-seed runs
        self.log_code()
        self.response_table = wandb.Table(
            columns=["objective-evaluations", "idx", "reward", "response"],
            log_mode="INCREMENTAL",
        )
        self.best_reward = -math.inf

    def set_seed(self):
        """Seeds Python's random, NumPy and torch with config.seed."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def setup_wandb(self):
        wandb.init(project="guide-language-diffusion", name=self.config.run_name, config=self.config.to_dict())

    def setup_task(self):
        self.task = tasks.get_reward_fn(self.config.task.name, concurrency=self.config.task.concurrency)

    def setup_pipeline(self, **kwargs):
        """Builds config.model's SGLang pipeline; kwargs go to sglang.Engine."""
        cfg = self.config.model
        self.pipeline = Pipeline(cfg.name, cfg.max_total_tokens, enable_thinking=cfg.enable_thinking, reasoning_effort=cfg.reasoning_effort, seed=self.config.seed, **kwargs)
        self.task.pipeline = self.pipeline

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
        rewards_finite = rewards[rewards.isfinite()]
        reward_best = rewards_finite.max().item() if len(rewards_finite) else -math.inf
        self.best_reward = max(self.best_reward, reward_best)
        log_dict = {
            "objective-evaluations": objective_evaluations,
            "rewards": rewards_finite.mean().item(),
            "reward-finite-ratio": len(rewards_finite) / len(rewards),
            "rewards-best": reward_best,
            "rewards-best-so-far": self.best_reward,
            **extra,
        }
        wandb.log(log_dict)

    def log_responses(self, objective_evaluations, rewards, responses, extra={}):
        """Logs each response with its reward to the incremental "responses" table.

        Args:
            objective_evaluations: int
            rewards: (N) float64
            responses: list (N) of str
            extra: dict, further entries of the same wandb.log
        """
        for idx, (response, reward) in enumerate(zip(responses, rewards)):
            self.response_table.add_data(objective_evaluations, idx, reward.item(), response)
        log_dict = {
            "objective-evaluations": objective_evaluations,
            "responses": self.response_table,
            **extra,
        }
        wandb.log(log_dict)
