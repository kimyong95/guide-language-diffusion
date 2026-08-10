import ml_collections

def get_config(algorithm="grpo"):
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.algorithm = algorithm
    config.run_name = algorithm

    config.max_epochs = 1000

    config.model = "Qwen/Qwen3-8B"
    config.task = "dapo-math-17k"

    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 512
    config.sample.m = 32
    config.sample.max_batch_size_per_device = 16
    config.sample.max_new_tokens = 20480
    config.sample.enable_thinking = True
    config.sample.temperature = 1.0

    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 3e-6
    config.train.max_grad_norm = 1.0
    config.train.gradient_checkpointing = True

    if algorithm == "grpo":
        config.train.beta = 0.001
        config.train.clip_ratio_low = 0.2
        config.train.clip_ratio_high = 0.2
    elif algorithm == "dr.grpo":
        config.train.clip_ratio_low = 0.2
        config.train.clip_ratio_high = 0.28
    elif algorithm == "nft":
        config.train.epsilon = 1.0
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    config.lora = ml_collections.ConfigDict()
    config.lora.r = 64
    config.lora.lora_alpha = 32
    config.lora.target_modules = "all-linear"

    return config
