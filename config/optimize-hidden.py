import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "optimize-hidden"

    config.max_epochs = 1000

    config.model = "Qwen/Qwen3-8B"
    config.task = "gsm8k"

    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 320     # N: total rollouts per epoch across all GPUs
    config.sample.m = 64                  # unique questions per epoch; group size k = N/m = 5
    config.sample.max_batch_size_per_device = 16
    config.sample.max_new_tokens = 4096
    config.sample.enable_thinking = False
    config.sample.n_intervene = 8 
    config.sample.temperature = 1.0

    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 0.03
    config.train.gradient_checkpointing = True

    return config
