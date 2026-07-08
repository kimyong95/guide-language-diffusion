import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "open-evolve"

    config.total_iterations = 100  # total codes generated/evaluated (one per iteration)

    config.model = "google/diffusiongemma-26B-A4B-it"
    config.task = "circle-packing"

    config.sample = ml_collections.ConfigDict()
    config.sample.num_inference_steps = 48
    config.sample.max_tokens = 262144    # token budget; runs max_tokens // canvas_length blocks
    config.sample.canvas_length = 256
    config.sample.entropy_bound = 0.1
    config.sample.t_min = 0.4
    config.sample.t_max = 0.8
    config.sample.enable_thinking = False

    # evolutionary search: MAP-Elites quality-diversity archive + island model (see openevolve-algorithm.md)
    config.num_islands = 5
    config.archive_size = 20        # elite pool for exploitation sampling
    config.num_inspirations = 3     # codes shown to the model besides the parent
    config.exploration_ratio = 0.2  # parent = explore(uniform) / exploit(elite) / weighted(remainder)
    config.exploitation_ratio = 0.7
    config.migration_interval = 20  # iterations between ring migrations
    config.migration_rate = 0.1     # top fraction of each island copied to ring neighbours

    return config

def get_config(name):
    return globals()[name]()
