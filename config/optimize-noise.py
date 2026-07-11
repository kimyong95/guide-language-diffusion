import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "optimize-noise"

    config.max_epochs = 100

    config.model = "google/diffusiongemma-26B-A4B-it"
    config.task = "circle-packing"

    # total objective evaluations: 100*8=800 (max_epochs * sample.total_samples)
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 16
    config.sample.num_inference_steps = 48
    config.sample.max_blocks = 1024
    config.sample.canvas_length = 256
    config.sample.entropy_bound = 0.1
    config.sample.t_min = 0.4
    config.sample.t_max = 0.8
    config.sample.enable_thinking = False
    config.sample.noise_length = 8  # model-noise entries appended per full_attention layer

    config.lr = 1.0  # (mu, sigma) update step size

    return config

def get_config(name):
    return globals()[name]()
