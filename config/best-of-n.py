import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "best-of-n"

    config.max_epochs = 100

    config.model = "google/diffusiongemma-26B-A4B-it"
    config.task = "circle-packing"

    config.top_k = 3  # best-program archive shown in the prompt (distinct rewards, best-first)

    # total objective evaluations: 100*8=800 (max_epochs * sample.total_samples)
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 8
    config.sample.num_inference_steps = 48
    config.sample.max_tokens = 262144
    config.sample.gen_length = 256
    config.sample.entropy_bound = 0.1
    config.sample.t_min = 0.4
    config.sample.t_max = 0.8
    config.sample.enable_thinking = False

    return config

def get_config(name):
    return globals()[name]()
