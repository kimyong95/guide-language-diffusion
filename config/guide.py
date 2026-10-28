import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "guide"
    config.task = "circle-packing-26"
    config.output_dir = "checkpoints/guide"

    config.L = 100
    config.N = 5
    config.algo_mode = "A"  # or "B"

    config.model = ml_collections.ConfigDict()
    config.model.name = "Qwen/Qwen3.8-27B-FP8"
    config.model.max_total_tokens = 100000
    config.model.enable_thinking = True
    config.model.reasoning_effort = "medium"

    return config
