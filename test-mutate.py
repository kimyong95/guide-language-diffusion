"""Mutates Qwen3.5-4B's stories toward CRITERIA by sampling under a reward-weighted product of mutation guides, no training.

Stage 0 samples N_SAMPLES stories from MODEL under STORY_PROMPT, which form the buffer. Each of N_ITERATIONS steps then
samples N_SAMPLES new stories independently, each under Σ_i w_i log p_i over the buffer's stories i, renormalised at
every token, under SAMPLING_CONFIG; p_i is MODEL under MUTATE_PROMPT holding story i, scoring the shared sampled story
after its own prompt. The weights anchor on --anchor, "weight" z⁺_i = (r_i - r_min) / Σ_j (r_j - r_min), "best"
1/|B| on the stories B tied for the top reward, or "mean" 1/N, and add the NFT weight (r_i - r̄) / (N r̄ (1 - r̄))
with --guided, which sums to zero, so they sum to 1; all rewards tied gives 1/N each. A verifier scores each story by
asking MODEL every one of CRITERIA through VERIFIER_PROMPT, each criterion's score being p(yes) / (p(yes) + p(no)) at
the answer's first token, p(yes) summing "Yes" and "yes" and p(no) "No" and "no", for reward
r = Σ score / len(CRITERIA); the buffer then keeps the N_SAMPLES highest-reward stories among the new ones and itself, the
new ones first on ties. Under accelerate launch the samples are split across
processes, each holding the model on its own GPUs and every guide in one cache, sampling and verifying its own share;
the shares are gathered so every process keeps the same buffer. Each step prints its stories, answers and scores.
"""

import argparse

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed
from transformers import GenerationConfig

from pipeline import Pipeline, VarlenCache

MODEL = "Qwen/Qwen3.5-4B"
SAMPLING_CONFIG = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
N_SAMPLES = 8
N_ITERATIONS = 10
MAX_STORY_TOKENS = 4096
SEED = 0

parser = argparse.ArgumentParser()
parser.add_argument("--anchor", choices=["weight", "best", "mean"], default="weight")
parser.add_argument("--guided", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

STORY_PROMPT = "Write a short story."

MUTATE_PROMPT = """\
Write a similar story as the following, no longer than it:

{story}"""

VERIFIER_PROMPT = """\
Read the story and answer the question with Yes or No only.

Story:
{story}

Question: {question}"""

CRITERIA = [
    "Does the story feature any animal?",
    "Is an animal one of the main characters?",
    "Is the story cute or heartwarming?",
    "Is the main animal a mammal?",
    "Is the main animal furry?",
    "Is the main animal someone's pet?",
    "Does the main animal wag its tail, fetch, or chase a ball?",
    "Does the main animal bark?",
    "Is the main animal a canine, such as a dog, puppy, wolf or fox?",
    "Does the story have a dog?",
]


def guide_weights(rewards):
    """Weighs each buffer story's guide by args.anchor, adding the NFT weight when args.guided.

    Args:
        rewards: list (N) of float in [0, 1]

    Returns:
        list (N) of float summing to 1. args.anchor "weight" gives z⁺_i = (r_i - r_min) / Σ_j (r_j - r_min), "best"
        1/|B| on the stories B tied for the top reward, "mean" 1/N; args.guided adds (r_i - r̄) / (N r̄ (1 - r̄)), which
        sums to zero. 1/N each when all rewards tie.
    """
    rewards = torch.tensor(rewards)
    if rewards.min() == rewards.max():
        return [1 / len(rewards)] * len(rewards)
    weights = {
        "weight": (rewards - rewards.min()) / (rewards - rewards.min()).sum(),
        "best": (rewards == rewards.max()) / (rewards == rewards.max()).sum(),
        "mean": torch.full_like(rewards, 1 / len(rewards)),
    }[args.anchor]
    if args.guided:
        weights = weights + (rewards - rewards.mean()) / (len(rewards) * rewards.mean() * (1 - rewards.mean()))
    return weights.tolist()


def next_log_probs(cache, n_labels):
    """Runs one forward over a cache keyed by (label, index) and ordered by label.

    Args:
        cache: VarlenCache
        n_labels: int, the cache's number of labels

    Returns:
        (n_labels, N_live, V) log probabilities
    """
    hidden = pipeline.model.get_decoder()(**cache.model_inputs()).last_hidden_state[0, cache.output_positions]
    return pipeline.model.get_output_embeddings()(hidden).float().log_softmax(-1).unflatten(0, (n_labels, -1))


@torch.no_grad()
def guided_generate(prompts, weights, n):
    """Samples n stories under Σ_i w_i log p_i, renormalised at every token, p_i being MODEL under prompts[i].

    Every guide holds every story in one cache, keyed by (label, index), the label being the guide.

    Args:
        prompts: list (G) of str, the guides' prompts
        weights: list (G) of float, the guides' w_i
        n: int, number of stories

    Returns:
        list (n) of str, sampled under SAMPLING_CONFIG's processors applied to the guided log probabilities
    """
    cache = VarlenCache({(label, index): tokens for label, tokens in enumerate(pipeline.texts_to_tokens(prompts)) for index in range(n)}, pipeline.device, pipeline.config)
    weights = torch.tensor(weights, device=pipeline.device)
    logits_processors = pipeline.model._get_logits_processor(SAMPLING_CONFIG, device=pipeline.device)
    generated_tokens = [[] for _ in range(n)]
    step = 0
    while cache.lengths and step < MAX_STORY_TOKENS:
        instance_ids = [index for label, index in cache.instance_ids if label == 0]
        log_probs_guided = (weights[:, None, None] * next_log_probs(cache, len(prompts))).sum(0)
        scores = logits_processors(torch.tensor([generated_tokens[i] for i in instance_ids], device=pipeline.device, dtype=torch.long), log_probs_guided)
        sampled = torch.multinomial(scores.softmax(-1), 1)[:, 0].tolist()
        for index, token in zip(instance_ids, sampled):
            generated_tokens[index].append(token)
        live = {index: token for index, token in zip(instance_ids, sampled) if token not in pipeline.eos_token_ids}
        cache.advance({(label, index): live[index] for label, index in cache.instance_ids if index in live})
        step += 1
    return pipeline.tokens_to_texts(generated_tokens)


@torch.no_grad()
def verify(stories):
    """Scores every criterion on every story from the verifier's first answer token.

    Every question is held in one cache, keyed by (label, index), the label being the story and the index the criterion.

    Args:
        stories: list (N) of str

    Returns:
        2D list (N, len(CRITERIA)) of float, p(yes) / (p(yes) + p(no)), p(yes) summing "Yes" and "yes", p(no) "No" and "no"
    """
    questions = {(i, j): VERIFIER_PROMPT.format(story=story, question=question) for i, story in enumerate(stories) for j, question in enumerate(CRITERIA)}
    cache = VarlenCache(dict(zip(questions, pipeline.texts_to_tokens(list(questions.values())))), pipeline.device, pipeline.config)
    log_probs = next_log_probs(cache, len(stories))
    log_probs_yes, log_probs_no = (log_probs[..., pipeline.tokenizer.convert_tokens_to_ids(words)].logsumexp(-1) for words in (["Yes", "yes"], ["No", "no"]))
    return (log_probs_yes - log_probs_no).sigmoid().tolist()


def sample(prompts, weights):
    """Samples and verifies this process's share of N_SAMPLES stories, gathered across processes.

    Args:
        prompts: list (G) of str, the guides' prompts
        weights: list (G) of float, the guides' w_i

    Returns:
        list (N_SAMPLES) of dict with "story", "scores", list (len(CRITERIA)) of float, and "reward"; the same on every
        process
    """
    n = len(range(accelerator.process_index, N_SAMPLES, accelerator.num_processes))
    stories = [story.strip() for story in guided_generate(prompts, weights, n)]
    return gather_object([{"story": story, "scores": scores, "reward": sum(scores) / len(CRITERIA)} for story, scores in zip(stories, verify(stories))])


accelerator = Accelerator()
set_seed(SEED, device_specific=True)
pipeline = Pipeline(MODEL, max_memory={i: torch.cuda.get_device_properties(i).total_memory for i in range(accelerator.process_index, torch.cuda.device_count(), accelerator.num_processes)})
buffer = []
for iteration in range(N_ITERATIONS + 1):
    if iteration == 0:
        prompts, weights = [STORY_PROMPT], [1.0]
    else:
        prompts = [MUTATE_PROMPT.format(story=data_i["story"]) for data_i in buffer]
        weights = guide_weights([data_i["reward"] for data_i in buffer])
    data = sample(prompts, weights)
    buffer = sorted(data + buffer, key=lambda data_i: data_i["reward"], reverse=True)[:N_SAMPLES]
    for i, data_i in enumerate(data):
        accelerator.print(f"{'=' * 30} iteration {iteration}, sample {i}, score {sum(data_i['scores']):.2f}/{len(CRITERIA)} {'=' * 30}")
        accelerator.print(data_i["story"])
        accelerator.print("-" * 80)
        for question, score in zip(CRITERIA, data_i["scores"]):
            accelerator.print(f"[{score:.2f}] {question}")
    scores = [sum(data_i["scores"]) for data_i in data]
    buffer_scores = [sum(data_i["scores"]) for data_i in buffer]
    accelerator.print(f"{'#' * 20} iteration {iteration}: mean score {sum(scores) / len(scores):.2f}/{len(CRITERIA)}, best {max(scores):.2f}, worst {min(scores):.2f}; "
                      f"buffer mean {sum(buffer_scores) / len(buffer_scores):.2f}, best {max(buffer_scores):.2f}, worst {min(buffer_scores):.2f}; "
                      f"weights {[round(weight, 3) for weight in weights]}")
    for j, question in enumerate(CRITERIA):
        accelerator.print(f"{sum(data_i['scores'][j] for data_i in data):.2f}/{len(data)} {question}")

accelerator.end_training()
