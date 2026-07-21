import math
import os
import sys
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        self.best_reward = {}

    def setup_accelerator(self):
        self.accelerator = Accelerator(log_with="wandb")
        self.accelerator.init_trackers(
            project_name="guide-language-diffusion",
            config=self.config,
            init_kwargs={"wandb": {"name": self.config.run_name, "config": self.config.to_dict()}}
        )
        set_seed(self.config.seed, device_specific=True)

    def setup_task(self):
        self.task = tasks.get_reward_fn(self.config.task)

    def setup_model(self):
        max_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(self.accelerator.process_index, torch.cuda.device_count(), self.accelerator.num_processes)}

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto", max_memory=max_memory)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def build_prompt_tokens(self, user_prompt, system_prompt="You are a helpful assistant.", enable_thinking=True):
        """Tokenize `user_prompt` into a batch-1 (1, P) input_ids tensor via the chat template,
        prepending `system_prompt` as a system turn."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=enable_thinking).to(self.model.device).input_ids  # (1, P)

    def strip_pads(self, tokens_list):
        """Drop the pad tokens from each entry of `tokens_list` (a (B, L) batch or list of rows) ->
        list of unpadded 1-D token rows."""
        return [tokens[tokens != self.tokenizer.pad_token_id] for tokens in tokens_list]

    def strip_eos(self, tokens_list):
        """Truncate each entry of `tokens_list` after its first eos; entries without an eos are kept
        whole -> list of 1-D token rows."""
        eos_token_ids = torch.tensor(self.model.generation_config.eos_token_id, device=self.model.device)

        stripped = []
        for tokens in tokens_list:
            eos_positions = torch.isin(tokens, eos_token_ids).nonzero(as_tuple=True)[0]
            end = eos_positions[0].item() + 1 if eos_positions.numel() > 0 else tokens.shape[0]
            stripped.append(tokens[:end])
        return stripped

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
