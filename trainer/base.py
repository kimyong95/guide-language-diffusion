import math
import os
import sys
from datetime import timedelta
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
import tasks
from pipeline import Pipeline


class BaseTrainer:

    def __init__(self, config):
        self.config = config
        self.setup_accelerator()
        self.setup_task()
        self.setup_pipeline()
        self.log_code()
        self.text_table = {
            "sampling": wandb.Table(
                columns=["objective-evaluations", "idx", "reward", "text"],
                log_mode="INCREMENTAL",
            ),
        }
        self.best_reward = {}

    def setup_accelerator(self):
        self.accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=30))],  # ranks skew by a whole sampling batch
        )
        self.accelerator.init_trackers(
            project_name="guide-language-diffusion",
            config=self.config,
            init_kwargs={"wandb": {"name": self.config.run_name, "config": self.config.to_dict()}}
        )
        set_seed(self.config.seed, device_specific=True)

    def setup_task(self):
        self.task = tasks.get_reward_fn(self.config.task)

    def setup_pipeline(self):
        """Give each rank its own slice of the visible GPUs, so G ranks never contend for one device."""
        max_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(self.accelerator.process_index, torch.cuda.device_count(), self.accelerator.num_processes)}
        self.pipeline = Pipeline(self.config.model, max_memory=max_memory)

    def log_code(self):
        if not self.accelerator.is_main_process:
            return

        cwd = os.path.abspath(os.getcwd())
        imported_py_files = set()
        for module in sys.modules.values():
            path = getattr(module, "__file__", None)
            if path and path.endswith(".py"):
                abs_path = os.path.abspath(path)
                if abs_path.startswith(cwd):
                    imported_py_files.add(abs_path)

        self.accelerator.get_tracker("wandb").run.log_code(".", include_fn=lambda path: path in imported_py_files)

    def log_rewards(self, objective_evaluations, rewards, stage, extra={}):
        self.best_reward[stage] = max(self.best_reward.get(stage, -math.inf), rewards.max().item())
        log_dict = {
            "objective-evaluations": objective_evaluations,
            f"{stage}/rewards": rewards.mean().item(),
            f"{stage}/rewards-best": rewards.max().item(),
            f"{stage}/best-so-far": self.best_reward[stage],
            **extra,
        }
        self.accelerator.log(log_dict)

    def log_texts(self, objective_evaluations, rewards, texts, stage, extra={}):
        if not self.accelerator.is_main_process:
            return
        table = self.text_table[stage]
        for idx, (text, reward) in enumerate(zip(texts, rewards)):
            table.add_data(objective_evaluations, idx, reward.item(), text)
        log_dict = {
            "objective-evaluations": objective_evaluations,
            f"{stage}/texts": table,
            **extra,
        }
        self.accelerator.get_tracker("wandb").log(log_dict)
