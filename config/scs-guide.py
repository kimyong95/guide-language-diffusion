import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "scs-guide"
    config.task = ml_collections.ConfigDict()
    config.task.name = "circle-packing-26"
    config.task.concurrency = 64

    config.L = 25   # number of steps
    config.K = 64   # number of new samples per run
    config.N = 5    # number of distributions
    config.E = 3    # number of in-context examples
    config.C = 8    # number of independent runs

    config.model = ml_collections.ConfigDict()
    config.model.name = "qwen-coder-next"
    config.model.max_total_tokens = 40000
    config.model.enable_thinking = False
    config.model.reasoning_effort = "medium"

    return config
