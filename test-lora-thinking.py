"""Fits a LoRA the model wears only while it is thinking, against one fixed ending.

The target is written once by the frozen model with thinking enabled, and what is kept is everything
from </think> onward: the reasoning that produced it is printed for reference but never scored, while
closing the block is, so the adapter is asked where to stop as well as what to say. The adapters are
worn from <think> through the last thinking token and nowhere else. Every step then lets the model
think again with the adapters on, BATCH_SIZE traces in one packed pass, and scores the target against
each block those samples produced, which asks how the reasoning has to be read for the target to
follow it. The batch's gradients accumulate into a single update, and reporting
follows the first trace. Each step also scores the block under the frozen model, and the two scorings
meet in the objective as a per-token KL: the adapter pays BETA nats for every nat the block moves away
from reasoning the frozen model would have written. Left unpriced, the adapter reaches the target by
collapsing the block into repetition and smuggling the story through the hidden states instead, and
repetition is the text a language model finds most likely, so the frozen log likelihood printed beside
the loss goes up rather than down and does not catch it on its own.
pipeline.lora_thinking finds each block from the token ids as they are
produced, keying its flags by request so traces that open and close at different steps are each
adapted over their own block, and so the prompt, both markers and the answer stay with the frozen
model. Gradients reach the adapters only through the thinking positions of the scoring forwards, never
through sampling, which runs under no_grad.
"""

import torch

from pipeline import Pipeline

MODEL = "Qwen/Qwen3-1.7B"
PROMPT = "讲一个故事,一个句子"
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 10    # thinking traces per step; their gradients are accumulated into one update
RANK = 8
ALPHA = 16
TEMPERATURE = 1.0
OPTIMIZE_STEPS = 100
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
BETA = 0.0    # per-token price on KL(adapted ‖ frozen) over the block; both terms are means, so one nat of drift trades against one nat of target
PRINT_CHARS = 100    # the in-loop thinking and story are only glanced at, so they are cut to this
SEED = 0


def clip(text):
    """Returns text cut to PRINT_CHARS characters, saying how much was left off."""
    return text if len(text) <= PRINT_CHARS else f"{text[:PRINT_CHARS]}… (+{len(text) - PRINT_CHARS} chars)"


def print_text(title, text):
    print("=" * 80)
    print(title)
    print(text)


def generate(pipeline, prompt_tokens, batch_size):
    """Samples with the adapters on and returns separate context and response token lists.

    The whole batch decodes in one packed pass, since lora_thinking keys its flags by request and so
    gates each trace over its own block however far apart they open and close.

    Args:
        pipeline: Pipeline, with init_thinking_lora already called.
        prompt_tokens: list (Lp), templated with enable_thinking=True, so the assistant turn is still
            open and the model writes <think> itself.
        batch_size: How many traces to sample from the one prompt; sampling alone separates them.

    Returns:
        A pair of lists (contexts, responses), each containing batch_size token lists. Each context is
        prompt + <think> + thinking, ending at the last thinking token, which is the last position the
        adapters are worn on; each response is everything from </think> onward, which is what a loss
        scores. A block the token cap cut short simply ends unclosed, and its response is empty.
    """
    close_id = pipeline.tokenizer.convert_tokens_to_ids("</think>")
    with pipeline.lora_thinking():
        batch = pipeline.generate([prompt_tokens] * batch_size, max_new_tokens=MAX_NEW_TOKENS).tokens
    contexts, responses = [], []
    for generated in batch:
        close = generated.index(close_id) if close_id in generated else len(generated)
        contexts.append(prompt_tokens + generated[:close])
        responses.append(generated[close:])
    return contexts, responses


torch.manual_seed(SEED)
pipeline = Pipeline(MODEL, temperature=TEMPERATURE)
lora_parameters = pipeline.init_thinking_lora(rank=RANK, alpha=ALPHA)

prompt_tokens = pipeline.texts_to_tokens([PROMPT], enable_thinking=True)[0]    # assistant turn still open, so the model writes <think> itself
open_id, close_id = pipeline.tokenizer.convert_tokens_to_ids(["<think>", "</think>"])

generated = pipeline.generate([prompt_tokens], max_new_tokens=MAX_NEW_TOKENS).tokens[0]    # no context open, so the adapters are inert and this is the frozen model
assert open_id in generated, "the model did not open a block, so the gate would stay shut and train nothing"
assert close_id in generated, f"the target's block did not close within {MAX_NEW_TOKENS} tokens"
close = generated.index(close_id)
target = generated[close:]    # </think> onward, so closing the block is scored along with the story
assert len(target) > 1, f"the target's block used all {MAX_NEW_TOKENS} tokens and left no story"

print_text(f"TARGET THINKING ({close} tokens)", pipeline.tokenizer.decode(generated[:close]))
print_text(f"TARGET ({len(target)} tokens)", pipeline.tokenizer.decode(target))
print("=" * 80)
print(f"LORA   rank {RANK}, alpha {ALPHA}, {sum(p.numel() for p in lora_parameters):,} parameters over {len(lora_parameters)} tensors, thinking tokens only")

optimizer = torch.optim.Adam(lora_parameters, lr=LEARNING_RATE)
for step in range(1, OPTIMIZE_STEPS + 1):
    contexts, responses = generate(pipeline, prompt_tokens, BATCH_SIZE)
    optimizer.zero_grad()
    nlls, kls, thinking_sums, thinking_means = [], [], [], []
    for context in contexts:    # one scored forward per trace, since log_probs takes a single sequence
        thinking = context[len(prompt_tokens):]    # <think> through the last thinking token, exactly the span the adapters are worn on
        with pipeline.lora_thinking():    # block and target in one pass, so the two terms cost a single forward
            adapted = pipeline.log_probs(prompt_tokens, thinking + target)
        with torch.no_grad():    # no context open, so the adapters are inert and this is the frozen model reading the same trace
            frozen = pipeline.log_probs(prompt_tokens, thinking)
        nll = -adapted[-len(target):].mean()
        kl = (adapted[:len(thinking)] - frozen).mean()    # the block was sampled from the adapted model, so its own score minus the frozen one estimates KL(adapted ‖ frozen)
        ((nll + BETA * kl) / BATCH_SIZE).backward()    # accumulate, so the update sees the batch mean
        nlls.append(nll.item())
        kls.append(kl.item())
        thinking_sums.append(frozen.sum().item())
        thinking_means.append(frozen.mean().item())
    grad_norm = torch.nn.utils.clip_grad_norm_(lora_parameters, GRAD_CLIP)
    assert torch.isfinite(grad_norm), f"step {step}: non-finite gradient, nlls = {nlls}, kls = {kls}"
    optimizer.step()
    context, response = contexts[0], responses[0]    # reporting follows the first trace only
    nll, kl = sum(nlls) / BATCH_SIZE, sum(kls) / BATCH_SIZE
    print(f"  step {step:3d}/{OPTIMIZE_STEPS}   loss = {nll + BETA * kl:.4f}   nll {nll:.4f}   kl {kl:.4f}   grad {grad_norm.item():.3f}   block {len(context) - len(prompt_tokens)} tokens")
    print(f"    thinking logp (frozen)   sum {sum(thinking_sums) / BATCH_SIZE:8.1f}   avg {sum(thinking_means) / BATCH_SIZE:.4f}")
    print_text("THINKING", clip(pipeline.tokenizer.decode(context[len(prompt_tokens):])))
    print_text("STORY", clip(pipeline.tokens_to_texts([response])[0]))

with pipeline.lora_thinking():    # end to end: the model thinks wearing the adapters, then answers as itself
    print_text("SAMPLE", pipeline.generate([prompt_tokens], max_new_tokens=MAX_NEW_TOKENS).texts[0])
