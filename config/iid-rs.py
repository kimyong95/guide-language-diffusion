import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "iid-rs"
    config.task = ml_collections.ConfigDict()
    config.task.name = "circle-packing"
    config.task.concurrency = 64

    config.M = 200  # number of samples

    config.model = ml_collections.ConfigDict()
    config.model.name = "qwen-next-instruct"
    config.model.max_new_tokens = 8192
    config.model.enable_thinking = False
    config.model.reasoning_effort = "medium"

    return config
