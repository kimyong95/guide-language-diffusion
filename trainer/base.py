import os
import sys
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from pipeline import DiffusionGemmaPipeline
import tasks


class BaseTrainer:

    def __init__(self, config):
        self.config = config
        self.setup_accelerator()
        self.setup_task()
        self.setup_model()
        self.log_code()
        self.text_table = {
            "sampling": wandb.Table(
                columns=["objective-evaluations", "idx", "reward", "text"],
                log_mode="INCREMENTAL",
            ),
        }

    def setup_accelerator(self):
        self.accelerator = Accelerator(log_with="wandb")
        self.accelerator.init_trackers(
            project_name="guide-language-diffusion",
            config=self.config,
            init_kwargs={"wandb": {"name": self.config.run_name, "config": self.config.to_dict()}}
        )
        set_seed(self.config.seed, device_specific=True)
        # assert torch.cuda.device_count() == self.accelerator.num_processes, f"Number of avaliable GPUs does not match the number of processes ({self.accelerator.num_processes})"
        assert self.config.sample.total_samples % self.accelerator.num_processes == 0, "total_samples must be divisible by num GPUs"

    def setup_task(self):
        self.task = tasks.get_reward_fn(self.config.task)

    def setup_model(self):
        self.pipeline = DiffusionGemmaPipeline(
            self.config.model,
            gen_length=self.config.sample.gen_length,
            entropy_bound=self.config.sample.entropy_bound,
            t_min=self.config.sample.t_min,
            t_max=self.config.sample.t_max,
        )
        self.pipeline.model.requires_grad_(False)

    @torch.no_grad()
    def completion(self, prompt, x1_tokens, max_new_tokens=4096):
        """Generate the answer that follows an already-generated thinking canvas.

        Splices the thinking canvas after the same thinking-enabled chat prompt the pipeline
        encodes, then lets the official `model.generate` autoregressively write the answer.

        Args:
            prompt: the user question (str).
            x1_tokens: (L,) long thinking canvas.
        Returns:
            decoded thinking + answer text (special tokens kept).
        """
        prompt_ids = self.pipeline.processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )["input_ids"].to(self.pipeline.input_device)  # (1, P)
        P = prompt_ids.shape[1]

        input_ids = torch.cat([prompt_ids, x1_tokens[None]], dim=-1)  # (1, P+L)
        output = self.pipeline.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        return self.pipeline.tokens_to_text(output.sequences[0, P:])  # thinking + answer

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
        log_dict = {
            "objective-evaluations": objective_evaluations,
            f"{stage}/rewards": rewards.mean().item(),
            f"{stage}/rewards-best": rewards.max().item(),
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
