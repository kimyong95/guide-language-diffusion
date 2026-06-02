import ml_collections

def base():
    config = ml_collections.ConfigDict()

    config.seed = 0
    config.run_name = "flow-guide"

    config.max_epochs = 100

    config.model = "GSAI-ML/LLaDA-8B-Instruct"
    config.task = "sudoku:0"

    # TODO(text): guidance hyperparameters

    # total objective evaluations: 100*16=1600
    config.sample = ml_collections.ConfigDict()
    config.sample.total_samples = 8
    config.sample.num_inference_steps = 100
    config.sample.gen_length = 100
    config.sample.temperature = 0.5

    return config

def get_config(name):
    return globals()[name]()
