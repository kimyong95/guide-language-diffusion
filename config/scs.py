import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "scs"
    config.task = ml_collections.ConfigDict()
    config.task.name = "circle-packing"
    config.task.concurrency = 64

    config.L = 10   # number of steps
    config.K = 20   # number of new samples per step
    config.E = 3    # number of in-context examples
    config.selection_policy = "random"  # "random", "best" or "best+random"

    config.model = ml_collections.ConfigDict()
    config.model.name = "qwen-next-instruct"
    config.model.max_new_tokens = 8192
    config.model.enable_thinking = False
    config.model.reasoning_effort = "medium"

    return config
