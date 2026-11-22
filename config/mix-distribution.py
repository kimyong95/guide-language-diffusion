import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "mix-distribution"
    config.task = ml_collections.ConfigDict()
    config.task.name = "story"
    config.task.concurrency = 32

    config.L = 1000  # number of steps
    config.K = 100   # number of new samples per step
    config.N = 5     # number of distributions
    config.E = 3     # number of in-context examples per distribution
    config.selection_policy = "best"  # "best" or "random"
    config.drop_policy = "worst"      # "worst" or "random"

    config.model = ml_collections.ConfigDict()
    config.model.name = "qwen3.5-4b"
    config.model.max_new_tokens = 8192
    config.model.enable_thinking = False
    config.model.reasoning_effort = None

    return config
