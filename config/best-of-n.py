import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "best-of-n"

    config.max_epochs = 100

    config.model = "Qwen/Qwen3-14B"
    config.task = "circle-packing"

    # total objective evaluations: 100*8=800 (max_epochs * sample.total_samples)
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 8
    config.sample.max_new_tokens = 8192
    config.sample.enable_thinking = True

    return config

def get_config(name):
    return globals()[name]()
