import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "open-evolve"

    config.max_epochs = 100

    config.model = "google/diffusiongemma-26B-A4B-it"
    config.task = "func-min"
    config.tp_size = 1

    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 8
    config.sample.num_inference_steps = 48
    config.sample.max_blocks = 16
    config.sample.gen_length = 256
    config.sample.entropy_bound = 0.1
    config.sample.t_min = 0.4
    config.sample.t_max = 0.8

    # evolutionary search
    config.archive_size = 20        # elite pool size for exploitation sampling
    config.num_inspirations = 3     # top programs shown in the prompt history
    config.exploitation_ratio = 0.7  # P(parent drawn from elite archive) else uniform-random

    return config

def get_config(name):
    return globals()[name]()
