import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "guide"
    config.task = ml_collections.ConfigDict()
    config.task.name = "circle-packing-26"
    config.task.concurrency = 64

    config.L = 100
    config.N = 5
    config.G = 5
    config.M = 3

    config.model = ml_collections.ConfigDict()
    config.model.name = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    config.model.max_total_tokens = 100000
    config.model.enable_thinking = False
    config.model.reasoning_effort = "medium"

    return config
