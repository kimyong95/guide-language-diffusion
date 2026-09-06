"""Fits one shared hidden-state rotation at prompt-text positions before every decoder layer."""

import torch

from pipeline import Pipeline

MODEL = "Qwen/Qwen3-1.7B"
PROMPT = "讲一个故事"    # no trailing 。: it merges with the echo instruction's newlines and shifts the turned span
ECHO_INSTRUCTION = "\n\nEcho the above prompt literally."    # sits in the same user turn, after PROMPT, and is never turned
MAX_NEW_TOKENS = 4096
TEMPERATURE = 1.0
OPTIMIZE_STEPS = 100
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0


@torch.no_grad()
def project_to_rotation(R):
    """Approximately orthogonalizes a near-rotation in float32 with three Newton–Schulz steps.

    Assumes positive determinant and singular values near one; does not correct reflections.
    """
    X = R.float()
    I = torch.eye(R.shape[0], device=R.device, dtype=torch.float32)
    for _ in range(3):
        X = 0.5 * X @ (3 * I - X.T @ X)
    return X


def rotation_angle(R):
    """Returns arccos(trace(R) / hidden_size) in degrees, as a diagnostic."""
    return torch.rad2deg(torch.arccos(torch.diagonal(R.double()).mean().clamp(-1, 1))).item()


def print_text(title, text):
    print("=" * 80)
    print(title)
    print(text)


def prompt_span(pipeline, suffix=""):
    """Returns chat tokens and the span overlapping PROMPT, including boundary-straddling tokens."""
    tokens = pipeline.texts_to_tokens([PROMPT + suffix])[0]
    text = pipeline.tokenizer.decode(tokens)
    encoding = pipeline.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    assert encoding.input_ids == tokens, "re-encoding the templated text must reproduce it token for token"
    first = text.index(PROMPT)
    start = next(i for i, (_, stop) in enumerate(encoding.offset_mapping) if stop > first)
    end = next(i for i, (begin, _) in enumerate(encoding.offset_mapping) if begin >= first + len(PROMPT))
    return tokens, (start, end)


pipeline = Pipeline(MODEL, temperature=TEMPERATURE)
D = pipeline.config.hidden_size

prompt_tokens, span = prompt_span(pipeline)
target = pipeline.generate([prompt_tokens], max_new_tokens=MAX_NEW_TOKENS).tokens[0]
print_text(f"TARGET ({len(target)} tokens)", pipeline.tokens_to_texts([target])[0])

print("=" * 80)
print(f"ROTATE   one {D}x{D} hidden-state rotation shared across all decoder layers, {span[1] - span[0]} prompt tokens, 1 target of {len(target)} tokens")
print(f"  turned span = {pipeline.tokenizer.decode(prompt_tokens[span[0]:span[1]])!r}")

R = torch.eye(D, device=pipeline.device, dtype=torch.float32).requires_grad_(True)    # Identity leaves the model unchanged.
optimizer = torch.optim.Adam([R], lr=LEARNING_RATE)
for step in range(1, OPTIMIZE_STEPS + 1):
    optimizer.zero_grad()
    with pipeline.rotate(R, [prompt_tokens], [span]) as [rotate_prompt_tokens]:
        loss = -pipeline.log_probs(rotate_prompt_tokens, target).mean()
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([R], GRAD_CLIP)
    assert torch.isfinite(grad_norm), f"step {step}: non-finite gradient, loss = {loss.item()}"
    optimizer.step()
    with torch.no_grad():
        R.copy_(project_to_rotation(R))
        angle = rotation_angle(R)
    print(f"  step {step:3d}/{OPTIMIZE_STEPS}   loss = {loss.item():.4f}   grad {grad_norm.item():.3f}   angle {angle:.2f}°")
R_hat = R.detach()

echo_tokens, echo_span = prompt_span(pipeline, ECHO_INSTRUCTION)
with pipeline.rotate(R_hat, [prompt_tokens, echo_tokens], [span, echo_span]) as rotate_prompts:
    sample, echo = pipeline.generate(rotate_prompts, max_new_tokens=MAX_NEW_TOKENS).texts
print_text("SAMPLE", sample)
print_text("ECHO", echo)
