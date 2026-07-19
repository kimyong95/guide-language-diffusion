import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "grpo"

    config.max_epochs = 100

    config.model = "Qwen/Qwen3-14B"
    config.task = "gsm8k"

    # total objective evaluations: 1000*40=40000 (max_epochs * sample.total_samples)
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 64
    config.sample.m = 8
    config.sample.max_batch_size_per_device = 16
    config.sample.max_new_tokens = 4096
    config.sample.enable_thinking = False

    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 1e-5
    config.train.beta = 0.04
    config.train.clip_range = 0.1
    config.train.gradient_updates_per_epoch = 1
    config.train.max_grad_norm = 1.0
    config.train.gradient_checkpointing = True

    config.lora = ml_collections.ConfigDict()
    config.lora.r = 16
    config.lora.lora_alpha = 32
    config.lora.target_modules = "all-linear"

    return config
