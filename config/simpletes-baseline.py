import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "simpletes"
    config.task = ml_collections.ConfigDict()
    config.task.name = "circle-packing-26"
    config.task.concurrency = 64

    config.L = 100
    config.C = 32
    config.K = 16
    config.N = 5

    config.model = ml_collections.ConfigDict()
    config.model.name = "qwen-coder-next"
    config.model.max_total_tokens = 100000
    config.model.enable_thinking = False
    config.model.reasoning_effort = "medium"

    return config
