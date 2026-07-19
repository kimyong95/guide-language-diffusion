import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "optimize-hidden"

    config.max_epochs = 100

    config.model = "Qwen/Qwen3-8B"
    config.task = "gsm8k"

    # total objective evaluations: 100*16=1600 (max_epochs * sample.total_samples)
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 16     # B: total rollouts per epoch across all GPUs
    config.sample.m = 4                  # unique questions per epoch; group size (generation batch) k = B/m = 8
    config.sample.max_new_tokens = 4096  # bounds the teacher-forced backward pass -- no gradient checkpointing with an injected KV cache
    config.sample.enable_thinking = False
    config.sample.noise_length = 8   # per-layer count L of injected KV rows; total optimized dims H*L*D
    config.sample.temperature = 1.0  # rollouts must be stochastic: x is one deterministic parameter, so greedy
    config.sample.top_p = 1.0        # decoding would give N identical samples and zero advantage spread

    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 0.03

    return config
