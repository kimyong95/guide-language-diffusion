"""One rotation of every layer's attention values at the prompt text, fitted to one continuation of that prompt.

The model writes a single target from PROMPT. It is then read back on the same prompt, but inside every
decoder layer the value each head computes at the tokens of PROMPT's own text is turned by one shared
rotation R before the heads read it. Queries and keys are left alone, so the attention pattern is the
model's own and only what it carries back from those positions is turned. Only that span is turned: the
chat template around it, the assistant header, and every generated token are left alone, so the turn reaches
what follows only through the values of those few positions. The prompt text never changes and the base
weights are frozen, so whatever R settles on is the whole intervention.

R is fitted off-policy, by the same likelihood test-reproduce.py's sft mode maximises,
    L(R) = -(1/|y|) sum_t log pi_R(y_t | y_<t)
over that one target, and after each step it is projected back onto the nearest rotation. At the end R is
read twice: one ordinary sample, and one with ECHO_INSTRUCTION appended inside the same user turn, asking
for the prompt back verbatim. The instruction itself is never turned, and the echo turns the same span and
nothing else. Sitting after PROMPT it leaves that span on the very positions it was fitted at, so the echo
changes nothing about the fit except what the sequence goes on to say.
"""

import torch

from pipeline import Pipeline

MODEL = "Qwen/Qwen3-1.7B"
PROMPT = "讲一个故事。"
ECHO_INSTRUCTION = "\n\nEcho the above prompt literally."    # sits in the same user turn, after PROMPT, and is never turned
MAX_NEW_TOKENS = 4096
TEMPERATURE = 1.0
OPTIMIZE_STEPS = 15
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
SEED = 0


def project_to_rotation(R):
    U, _, Vh = torch.linalg.svd(R)
    return U @ Vh


def rotation_angle(R):
    """
    Returns:
        float, degrees; how far R turns a direction on average, as the arccos of the mean cosine between a
        direction and its image, which over the sphere is trace(R) / Dv. 0 at the identity. One number for a
        turn that has Dv/2 principal angles, so it says how much R turns, not where.
    """
    return torch.rad2deg(torch.arccos((torch.diagonal(R).sum() / R.shape[0]).clamp(-1, 1))).item()


def print_text(title, text):
    print("=" * 80)
    print(title)
    print(text)


def prompt_span(pipeline, suffix=""):
    """
    Args:
        suffix: str, put in the user turn after PROMPT

    Returns:
        (list (Lp), (int, int)), the templated prompt and the half-open range of the tokens covering
        PROMPT's own characters, read off the offsets rather than diffed against the turn without it: the
        suffix follows PROMPT with no special token between them, so the tokenizer merges across that
        boundary and a straddling token is counted as PROMPT's
    """
    tokens = pipeline.texts_to_tokens([PROMPT + suffix])[0]
    text = pipeline.tokenizer.decode(tokens)
    encoding = pipeline.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    assert encoding.input_ids == tokens, "re-encoding the templated text must reproduce it token for token"
    first = text.index(PROMPT)
    start = next(i for i, (_, stop) in enumerate(encoding.offset_mapping) if stop > first)
    end = next(i for i, (begin, _) in enumerate(encoding.offset_mapping) if begin >= first + len(PROMPT))
    return tokens, (start, end)


pipeline = Pipeline(MODEL, temperature=TEMPERATURE)
Dv = pipeline.layers[0].self_attn.v_proj.out_features    # num_key_value_heads * head_dim, the whole concatenated value

prompt_tokens, span = prompt_span(pipeline)
target = pipeline.generate([prompt_tokens], max_new_tokens=MAX_NEW_TOKENS).tokens[0]
print_text(f"TARGET ({len(target)} tokens)", pipeline.tokens_to_texts([target])[0])

print("=" * 80)
print(f"ROTATE   one {Dv}x{Dv} value rotation over {span[1] - span[0]} of {len(prompt_tokens)} prompt tokens, 1 target of {len(target)} tokens")
print(f"  turned span = {pipeline.tokenizer.decode(prompt_tokens[span[0]:span[1]])!r}")

R = torch.eye(Dv, device=pipeline.device, dtype=torch.float32).requires_grad_(True)    # identity, so step 1's loss is the model untouched
optimizer = torch.optim.Adam([R], lr=LEARNING_RATE)
for step in range(1, OPTIMIZE_STEPS + 1):
    optimizer.zero_grad()
    with pipeline.rotate(R.to(pipeline.model.dtype), [prompt_tokens], [span]) as [rotate_prompt_tokens]:   # the cast is a graph node, and backward frees it each step
        loss = -pipeline.log_probs(rotate_prompt_tokens, target).mean()
        loss.backward()    # inside the block, so this step's graph is gone before the next forward
    grad_norm = torch.nn.utils.clip_grad_norm_([R], GRAD_CLIP)
    assert torch.isfinite(grad_norm), f"step {step}: non-finite gradient, loss = {loss.item()}"
    optimizer.step()
    with torch.no_grad():
        R.copy_(project_to_rotation(R))
        angle = rotation_angle(R)
    print(f"  step {step:3d}/{OPTIMIZE_STEPS}   loss = {loss.item():.4f}   grad {grad_norm.item():.3f}   angle {angle:.2f}°")
R_hat = R.detach().to(pipeline.model.dtype)

echo_tokens, echo_span = prompt_span(pipeline, ECHO_INSTRUCTION)
with pipeline.rotate(R_hat, [prompt_tokens, echo_tokens], [span, echo_span]) as rotate_prompts:
    sample, echo = pipeline.generate(rotate_prompts, max_new_tokens=MAX_NEW_TOKENS).texts
print_text("SAMPLE", sample)
print_text("ECHO", echo)
