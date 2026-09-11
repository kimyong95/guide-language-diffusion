"""Generates from a prompt whose visual slot holds random hidden states, and prints only the response.

Every layer reads its own random state at the slot.
"""

import math

import torch

from pipeline import Pipeline

MODEL = "Qwen/Qwen3.5-2B"
PROMPT = "Why the sky is blue?"
L = 128    # side of the square visual grid, so L * L slots
MAX_NEW_TOKENS = 4096
TEMPERATURE = 1.0
ENABLE_THINKING = False
SEED = 5


def project_to_sphere(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True) * math.sqrt(x.shape[-1])


pipeline = Pipeline(MODEL, temperature=TEMPERATURE)
n_layers = len(pipeline.layers)
D = pipeline.text_config.hidden_size
NUM_VISUAL = L * L

prompt_tokens = pipeline.texts_to_tokens([PROMPT], L=L, enable_thinking=ENABLE_THINKING)[0]
assert prompt_tokens.count(pipeline.visual_token_id) == NUM_VISUAL, "the slots must reach the prompt as image tokens, not as text"

hidden = project_to_sphere(torch.randn(n_layers, NUM_VISUAL, D, generator=torch.Generator().manual_seed(SEED))).to(pipeline.device)

torch.manual_seed(SEED)
with pipeline.inject_hidden(hidden):
    print(pipeline.generate([prompt_tokens], max_new_tokens=MAX_NEW_TOKENS).texts[0])
