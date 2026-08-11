import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "optimize-hidden"

    config.max_epochs = 1000

    config.model = ml_collections.ConfigDict()
    config.model.name = "Qwen/Qwen3-8B"
    config.model.max_batch_size_per_device = 16
    config.model.max_new_tokens = 20480
    config.model.enable_thinking = True
    config.model.temperature = 1.0
    config.model.n_intervene = 8

    config.sample = ml_collections.ConfigDict()
    config.sample.task = "dapo-math-17k"
    config.sample.total_samples = 512
    config.sample.m = 32 # number of distinct prompts

    config.val = ml_collections.ConfigDict()
    config.val.task = "aime-2024"
    config.val.every_n_epochs = 25
    config.val.k = 32 # Pass@K

    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 0.1
    config.train.epsilon = 1.0
    config.train.gradient_checkpointing = True

    return config
