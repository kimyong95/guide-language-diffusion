import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "optimize-hidden"

    config.max_epochs = 1000

    config.model = ml_collections.ConfigDict()
    config.model.name = "Qwen/Qwen3-1.7B"
    config.model.max_batch_size_per_device = 64
    config.model.max_new_tokens = 4096
    config.model.enable_thinking = False
    config.model.temperature = 1.0
    config.model.n_intervene = 8

    config.sample = ml_collections.ConfigDict()
    config.sample.task = "math-500"
    config.sample.total_samples = 512
    config.sample.m = 32 # number of distinct prompts

    config.val = ml_collections.ConfigDict()
    config.val.task = "aime-2024"
    config.val.every_n_epochs = 25
    config.val.k = 4 # Pass@K

    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 0.03
    config.train.epsilon = 1.0
    config.train.gradient_checkpointing = True

    return config
