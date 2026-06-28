import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "flow-guide"

    config.max_epochs = 100

    config.model = "google/diffusiongemma-26B-A4B-it"
    config.task = "sudoku:0"

    # total objective evaluations: 100*16=1600
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 8
    config.sample.num_inference_steps = 48
    config.sample.gen_length = 256
    config.sample.entropy_bound = 0.1
    config.sample.t_min = 0.4
    config.sample.t_max = 0.8

    config.guide_scale = 10
    config.guidance_layers = tuple(range(30))  # all 30 layers

    return config

def get_config(name):
    return globals()[name]()
