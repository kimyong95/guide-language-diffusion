from typing import List, Union

import einops
import torch
from accelerate.utils import gather_object
from accelerate.utils import gather as all_gather   # aliased: this file exports a gather of its own


def concat(data: Union[List[torch.Tensor], List[List]]):
    """Merge a list of per-batch values into one: cat tensors along dim 0, flatten lists
    (finetune-stable-diffusion/utils.py)."""
    if isinstance(data[0], torch.Tensor):
        return torch.cat(data, dim=0)
    elif isinstance(data[0], list):
        return sum(data, [])
    else:
        raise ValueError(f"Unsupported data type: {type(data[0])}")

def clamp_preserve_grad(input, min=None, max=None):
    """Clamp the forward value while keeping a non-zero gradient everywhere; a plain
    torch.clamp would zero the gradient of exactly the tokens the clamp is protecting."""
    return input + (input.clamp(min=min, max=max) - input).detach()

def batches_dict(data, batch_size):
    n = len(next(iter(data.values())))
    for i in range(0, n, batch_size):
        yield {k: v[i:i+batch_size] for k, v in data.items()}

def iter_dict(data):
    n = len(next(iter(data.values())))
    for i in range(n):
        yield {k: v[i] for k, v in data.items()}

def batch_slices(total, max_batch_size):
    """Partition ``range(total)`` into contiguous chunks of at most ``max_batch_size``.

    Yields ``slice`` objects so callers can read/assign one chunk at a time, e.g. to
    bound peak memory while keeping results identical to processing ``total`` at once.
    The final chunk may be smaller; ``max_batch_size`` need not divide ``total``.

    Args:
        total (int): number of items to split (e.g. a batch size).
        max_batch_size (int): maximum size of each chunk; must be >= 1.

    Yields:
        slice: contiguous slice covering one chunk, in order.
    """
    assert max_batch_size >= 1, "max_batch_size must be >= 1"
    for start in range(0, total, max_batch_size):
        yield slice(start, min(start + max_batch_size, total))

def gather(data):
    """All-gather that allows each process a different number of items, which the plain one
    cannot: its all_gather needs one shape on every process. The process count falls out of the
    size gather, so nothing has to be passed in.

    Args:
        data: (N_local, ...) tensor, or a length-N_local list

    Returns:
        (N, ...) tensor, or a length-N list; process-major, N = sum of the per-process counts
    """
    if not isinstance(data, torch.Tensor):
        return gather_object(data)

    sizes = all_gather(torch.tensor([len(data)], device=data.device)).tolist()   # one element per process, so the shapes match here
    padded = data.new_zeros(max(sizes), *data.shape[1:])
    padded[:len(data)] = data
    padded = einops.rearrange(all_gather(padded), "(num_processes N_max) ... -> num_processes N_max ...", num_processes=len(sizes))
    return torch.cat([block[:size] for block, size in zip(padded, sizes)])

def ungather(gathered, local_len, process_index):
    """Inverse of gather: hands a process back the block it put in. The per-process counts are
    recovered by a collective rather than assumed, so any assignment round-trips -- contiguous,
    strided, or uneven for reasons of its own.

    Args:
        gathered: (N, ...) tensor, or a length-N list; process-major, as gather returns it
        local_len (int): items this process put in
        process_index (int): this process
    """
    sizes = gather_object([local_len])
    start = sum(sizes[:process_index])
    return gathered[start:start + local_len]
