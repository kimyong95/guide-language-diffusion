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
