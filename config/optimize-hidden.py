import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "optimize-hidden"

    config.max_epochs = 100

    config.model = "Qwen/Qwen3-14B"
    config.task = "circle-packing"

    # total objective evaluations: 100*16=1600 (max_epochs * sample.total_samples)
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 16
    config.sample.max_new_tokens = 8192  # bounds the teacher-forced backward pass -- no gradient checkpointing with an injected KV cache
    config.sample.enable_thinking = True
    config.sample.noise_length = 8   # per-layer count L of injected KV rows; total optimized dims H*L*D
    config.sample.temperature = 1.0  # rollouts must be stochastic: x is one deterministic parameter, so greedy
    config.sample.top_p = 1.0        # decoding would give N identical samples and zero advantage spread

    config.train = ml_collections.ConfigDict()
    config.train.learning_rate = 1.0  # plain SGD; one clipped step moves at most lr in L2, ~1.4% of the sqrt(D) sphere radius
    config.train.max_grad_norm = 1.0  # clip the x gradient to L2 norm <= 1 before the SGD step
    config.train.adv_clip_max = 5.0   # clamp normalized advantages before they weight the loss

    return config

def get_config(name):
    return globals()[name]()
