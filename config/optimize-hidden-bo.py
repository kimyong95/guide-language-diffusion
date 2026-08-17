import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "optimize-hidden-bo"

    config.max_epochs = 1000

    config.model = ml_collections.ConfigDict()
    config.model.name = "Qwen/Qwen3-1.7B"
    config.model.max_batch_size_per_device = 64
    config.model.max_new_tokens = 4096
    config.model.enable_thinking = False
    config.model.temperature = 0.0 # x -> reward has to be a function, not a sample
    config.model.n_intervene = 1

    config.sample = ml_collections.ConfigDict()
    config.sample.task = "math-500" # y is the mean reward over the whole set, so num_processes must divide 500

    config.val = ml_collections.ConfigDict()
    config.val.task = "aime-2024"
    config.val.every_n_epochs = 25
    config.val.k = 4 # Pass@K
    config.val.temperature = 1.0 # pass@k needs a sampler, and sampling is greedy everywhere else

    config.bo = ml_collections.ConfigDict()
    config.bo.n_init = 10 # uniform-on-sphere draws before the GP takes over

    return config
