"""Perplexity evaluation: sliding-window NLL over a token stream."""

import numpy as np


def compute_perplexity(model, token_ids, block_size, stride=None):
    """Sliding-window BPE-level perplexity on a 1-D token sequence.

    token_ids: int ndarray, the corpus to evaluate.
    Returns float perplexity = exp(mean per-token NLL).
    """
    if stride is None:
        stride = block_size
    model.eval()
    nll_sum = 0.0
    n_tokens = 0
    for start in range(0, len(token_ids) - block_size, stride):
        idx = token_ids[start : start + block_size][None, :]
        tgt = token_ids[start + 1 : start + block_size + 1][None, :]
        _, loss = model(idx, tgt)
        # `cross_entropy(reduction="mean")` averages over the window's tokens;
        # multiply back so the across-window mean weights long/short evenly.
        nll_sum += float(loss.data) * block_size
        n_tokens += block_size
    model.train()
    return float(np.exp(nll_sum / n_tokens))
