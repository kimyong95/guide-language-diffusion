import ml_collections

def get_config(algorithm="grpo"):
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.algorithm = algorithm
    config.run_name = algorithm

    config.max_epochs = 1000

    config.model = ml_collections.ConfigDict()
    config.model.name = "Qwen/Qwen3-1.7B"
    config.model.max_batch_size_per_device = 16
    config.model.max_new_tokens = 4096
    config.model.enable_thinking = False
    config.model.temperature = 1.0

    config.sample = ml_collections.ConfigDict()
    config.sample.task = "dapo-math-17k"
    config.sample.total_samples = 512
    config.sample.m = 32 # number of distinct prompts

    config.val = ml_collections.ConfigDict()
    config.val.task = "aime-2024"
    config.val.every_n_epochs = 25
    config.val.k = 32 # Pass@K

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
