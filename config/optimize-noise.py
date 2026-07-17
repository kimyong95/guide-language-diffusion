import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "optimize-noise"

    config.max_epochs = 100

    config.model = "Qwen/Qwen3-14B"
    config.task = "circle-packing"

    # total objective evaluations: 100*16=1600 (max_epochs * sample.total_samples)
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 16
    config.sample.max_new_tokens = 8192  # per-rollout generation budget
    config.sample.enable_thinking = True
    config.sample.noise_length = 8  # per-layer count L of injected KV-noise rows; total optimized dims H*L*D

    config.lr = 1.0  # (mu, sigma) update step size

    return config

def get_config(name):
    return globals()[name]()
