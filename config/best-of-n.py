import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "best-of-n"

    config.max_epochs = 100

    config.model = "Qwen/Qwen3-8B"
    config.problem = "circle-packing"

    # total objective evaluations: 100*16=1600 (max_epochs * sample.total_samples)
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 16
    config.sample.max_batch_size_per_device = 8
    config.sample.max_new_tokens = 4096
    config.sample.enable_thinking = False
    config.sample.temperature = 1.0

    return config
