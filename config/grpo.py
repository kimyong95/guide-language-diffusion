import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "grpo"

    config.max_epochs = 1000

    config.model = "Qwen/Qwen3-8B"
    config.task = "gsm8k"

    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 320
    config.sample.m = 64
    config.sample.max_batch_size_per_device = 16
    config.sample.max_new_tokens = 4096
    config.sample.enable_thinking = False
    config.sample.temperature = 1.0

    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 3e-6
    config.train.beta = 0.001
    config.train.clip_range = 0.2
    config.train.gradient_updates_per_epoch = 2
    config.train.max_grad_norm = 1.0
    config.train.gradient_checkpointing = True

    config.lora = ml_collections.ConfigDict()
    config.lora.r = 64
    config.lora.lora_alpha = 32
    config.lora.target_modules = "all-linear"

    return config
