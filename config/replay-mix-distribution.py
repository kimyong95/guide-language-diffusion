import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "replay-mix-distribution"
    config.task = ml_collections.ConfigDict()
    config.task.name = "story"
    config.task.concurrency = 32

    config.L = 100  # number of steps
    config.M = 100  # number of new samples per step
    config.K = 5    # number of top distinct samples whose E-combinations are the candidates
    config.n = 5    # number of distributions in the mixture
    config.E = 3    # number of in-context examples per distribution

    config.model = ml_collections.ConfigDict()
    config.model.name = "qwen3.5-4b"
    config.model.max_new_tokens = 8192
    config.model.enable_thinking = False
    config.model.reasoning_effort = None

    return config
