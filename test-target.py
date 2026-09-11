"""Fits injected hidden states that make the model answer one prompt with one fixed target.

The slots are image placeholders opening the user turn, wrapped in <|vision_start|>/<|vision_end|>
exactly as a real picture would be, and every decoder layer reads its own fitted state there instead
of the empty placeholder. The model therefore reads them as the content of an image it was never
shown. Only those states are trained, so whatever it takes to turn the prompt into the target has to
fit inside L * L visual tokens, and the periodic samples show how far along that is.

Each of those samples is taken twice off the same states: once on PROMPT, which is what the fit
optimizes, and once on ECHO, which asks the model to describe the picture it thinks it is looking at.
Only the first is trained for, so the echo says what the fitted image reads as on a question it was
never fitted on -- whether the states carry something the model can put into words, or only a
shortcut to the target.
"""

import math

import torch

from pipeline import Pipeline

MODEL = "Qwen/Qwen3.5-2B"
PROMPT = "这张图片, 讲一个故事"
TARGET = "小熊和小比在森林里采蜜糖."
ECHO = "详细描述这张图片。"
L = 1
MAX_NEW_TOKENS = 128
OPTIMIZE_STEPS = 30
LEARNING_RATE = 1.0
GRAD_CLIP = 1.0
TEMPERATURE = 1.0
ENABLE_THINKING = False
PRINT_EVERY = 5
SEED = 1


def project_to_sphere(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True) * math.sqrt(x.shape[-1])


pipeline = Pipeline(MODEL, temperature=TEMPERATURE)
n_layers = len(pipeline.layers)
D = pipeline.text_config.hidden_size
torch.manual_seed(SEED)

prompt_tokens = pipeline.texts_to_tokens([PROMPT], L=L, enable_thinking=ENABLE_THINKING)[0]
echo_tokens = pipeline.texts_to_tokens([ECHO], L=L, enable_thinking=ENABLE_THINKING)[0]
target = pipeline.tokenizer(TARGET + pipeline.tokenizer.eos_token).input_ids    # the eos is fitted too, so a hit stops there instead of running on

hidden = project_to_sphere(torch.randn(n_layers, L * L, D, generator=torch.Generator().manual_seed(SEED))).to(pipeline.device, torch.float32).requires_grad_(True)
optimizer = torch.optim.Adam([hidden], lr=LEARNING_RATE)

for step in range(1, OPTIMIZE_STEPS + 1):
    optimizer.zero_grad()
    with pipeline.inject_hidden(hidden):
        loss = -pipeline.log_probs(prompt_tokens, target).mean()
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([hidden], GRAD_CLIP)
    assert torch.isfinite(grad_norm), f"step {step}: non-finite gradient, loss = {loss.item()}"
    optimizer.step()
    with torch.no_grad():
        hidden.copy_(project_to_sphere(hidden))

    print(f"step {step:3d}/{OPTIMIZE_STEPS}   loss = {loss.item():.4f}   grad {grad_norm.item():.3f}")
    if step == 1 or step % PRINT_EVERY == 0:
        with pipeline.inject_hidden(hidden.detach()):
            sample = pipeline.generate([prompt_tokens], max_new_tokens=MAX_NEW_TOKENS).texts[0]
            echo = pipeline.generate([echo_tokens], max_new_tokens=MAX_NEW_TOKENS).texts[0]
        print(f"sample at step {step}:")
        print(sample)
        print(f"echo at step {step}:")
        print(echo)
