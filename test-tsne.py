import os

import einops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

import tasks
from pipeline import LLaDAPipeline
from utils import batch_slices

# ---- config ----
MODEL = "GSAI-ML/LLaDA-8B-Instruct"
DEVICE = "cuda"
TASK = "sudoku:0"
N = 100
GEN_LENGTH = 128
NUM_INFERENCE_STEPS = 128
TEMPERATURE = 1.0          # >0 required, else all N samples are identical (greedy)
BATCH_SIZE = 10            # tune for GPU memory; does not affect results
CACHE_PATH = "test-tsne-cache.pt"
OUTPUT_DIR = "test-tsne-figures"
SEED = 0

pipeline = LLaDAPipeline(MODEL, device=DEVICE, gen_length=GEN_LENGTH, temperature=TEMPERATURE)


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS-normalization over the last (model) dimension."""
    return x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + eps)


def eos_ends(tokens):
    """Per-sentence kept length: first-eos index + 1 (include eos), else full L.

    Everything after the first eos (trailing pad/junk) is stripped.

    Args:
        tokens (N, L) long generated token ids
    Returns:
        ends (list[int], len N)
    """
    N_, L = tokens.shape
    ends = []
    for n in range(N_):
        eos_pos = (tokens[n] == pipeline.eos_token_id).nonzero().flatten()
        ends.append(int(eos_pos[0]) + 1 if len(eos_pos) else L)
    return ends


def eos_mean_pool(hidden, tokens):
    """Mean-pool each sentence over tokens up to & including the first eos.

    Args:
        hidden  (N, H, L, d) float hidden states
        tokens  (N, L)       long  generated token ids
    Returns:
        pooled  (N, H, d)
    """
    N_, H, L, d = hidden.shape
    pooled = torch.empty(N_, H, d)
    for n, end in enumerate(eos_ends(tokens)):
        pooled[n] = hidden[n, :, :end, :].mean(dim=1)
    return pooled


def generate_and_extract():
    """Generate N sentences, score them, and extract all-layer/all-token hidden states.

    Returns:
        tokens  (N, L)        long      generated token ids
        rewards (N,)          float     per-sentence task reward
        hidden  (N, H, L, d)  float16   rms-normed hidden states (CPU)
    """
    torch.manual_seed(SEED)
    task = tasks.get_reward_fn(TASK)
    prompt_tokens = pipeline.prompt_to_tokens(task.prompt())[0]  # (P,)

    # ---- generation (full-diffusion denoising from all-mask) ----
    token_chunks = []
    for sl in batch_slices(N, BATCH_SIZE):
        b = sl.stop - sl.start
        xt = pipeline.init_tokens(b)  # (b, L) all mask
        timesteps = pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
        for timestep in timesteps:
            logits, _ = pipeline.model_predict(xt, prompt_tokens)  # (b, L, V)
            xt = pipeline.scheduler.step(xt, logits, timestep)
        token_chunks.append(xt)
    tokens = torch.cat(token_chunks, dim=0)  # (N, L)

    # ---- rewards ----
    texts = pipeline.tokens_to_text(tokens)
    rewards = task.evaluate(texts)  # (N,)

    # ---- one forward pass per sample for all hidden states ----
    hidden_chunks = []
    for sl in batch_slices(N, BATCH_SIZE):
        _, hs = pipeline.model_predict(tokens[sl], prompt_tokens)  # (H, b, L, d)
        hs = rms_norm(hs.float())                                  # rms-norm each point over d
        hidden_chunks.append(einops.rearrange(hs, "H b L d -> b H L d").half().cpu())
    hidden = torch.cat(hidden_chunks, dim=0)  # (N, H, L, d)

    return tokens.cpu(), rewards.cpu(), hidden


def load_or_build():
    if os.path.exists(CACHE_PATH):
        print(f"loading cached data from {CACHE_PATH}")
        cache = torch.load(CACHE_PATH)
        return cache["tokens"], cache["rewards"], cache["hidden_states"]

    tokens, rewards, hidden = generate_and_extract()
    torch.save({"tokens": tokens, "rewards": rewards, "hidden_states": hidden}, CACHE_PATH)
    print(f"saved data to {CACHE_PATH}")
    return tokens, rewards, hidden


def tsne_scatter(points, colors, title, path):
    """Fit 2D t-SNE on raw points and save a reward-colored scatter as JPEG."""
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=SEED).fit_transform(points)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=colors, cmap="viridis", s=8)
    fig.colorbar(sc, ax=ax, label="reward")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, format="jpg", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokens, rewards, hidden = load_or_build()  # hidden: (N, H, L, d)

    N_, H, L, d = hidden.shape
    print(f"hidden states: N={N_} H={H} L={L} d={d}")
    print(f"rewards: min={rewards.min():.3f} max={rewards.max():.3f} mean={rewards.mean():.3f}")

    r = rewards.numpy()

    # ---- figure set 1: mean-pool over all L tokens, N points per layer ----
    pooled = hidden.float().mean(dim=2)  # (N, H, d)
    for h in range(H):
        tsne_scatter(
            pooled[:, h].numpy(),
            r,
            f"set1 mean-pool · layer {h}",
            os.path.join(OUTPUT_DIR, f"set1_meanpool_layer{h:02d}.jpg"),
        )
        print(f"[set1] layer {h} done")

    # ---- figure set 2: mean-pool over tokens up to & including eos, N points per layer ----
    pooled_eos = eos_mean_pool(hidden.float(), tokens)  # (N, H, d)
    for h in range(H):
        tsne_scatter(
            pooled_eos[:, h].numpy(),
            r,
            f"set2 mean-pool pre-eos · layer {h}",
            os.path.join(OUTPUT_DIR, f"set2_meanpool_eos_layer{h:02d}.jpg"),
        )
        print(f"[set2] layer {h} done")

    # ---- figure set 3: all tokens, N*L points per layer ----
    r_tok = rewards.repeat_interleave(L).numpy()  # (N*L,) aligns with (N,L)->(N*L) row-major
    for h in range(H):
        pts = hidden[:, h].reshape(N_ * L, d).float().numpy()  # (N*L, d)
        tsne_scatter(
            pts,
            r_tok,
            f"set3 all-tokens · layer {h}",
            os.path.join(OUTPUT_DIR, f"set3_alltokens_layer{h:02d}.jpg"),
        )
        print(f"[set3] layer {h} done")

    # ---- figure set 4: tokens up to & including eos, variable points per layer ----
    ends = eos_ends(tokens)
    r_tok_eos = torch.cat([rewards[n].repeat(ends[n]) for n in range(N_)]).numpy()  # (sum ends,)
    for h in range(H):
        pts = torch.cat([hidden[n, h, :ends[n], :].float() for n in range(N_)], dim=0).numpy()
        tsne_scatter(
            pts,
            r_tok_eos,
            f"set4 tokens pre-eos · layer {h}",
            os.path.join(OUTPUT_DIR, f"set4_alltokens_eos_layer{h:02d}.jpg"),
        )
        print(f"[set4] layer {h} done")

    print(f"saved {4 * H} figures to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
