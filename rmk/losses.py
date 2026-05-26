"""Loss functions. Hand-derived backward passes; no autograd shortcuts."""

from rmk.backend import xp
from rmk.tensor import Tensor


def cross_entropy(logits, targets, reduction="mean"):
    """Fused softmax + cross-entropy.

    logits   : Tensor of shape (..., C) — raw class scores.
    targets  : int ndarray of shape (...) — class indices in [0, C).
    reduction: "mean" or "sum" over all target positions.

    Backward gradient on logits collapses to (softmax(z) - one_hot(targets)) / N.
    Computed numerically stably via the log-sum-exp trick.
    """
    z = logits.data

    # numerically stable softmax probabilities
    shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = xp.exp(shifted)
    probs = exp_z / exp_z.sum(axis=-1, keepdims=True)

    # gather the probability assigned to the correct class for each position,
    # then compute negative log likelihood
    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_targets = targets.reshape(-1)
    N = flat_targets.size
    nll = -xp.log(flat_probs[xp.arange(N), flat_targets] + 1e-12)
    loss_val = nll.mean() if reduction == "mean" else nll.sum()

    out = Tensor(loss_val, (logits,), "cross_entropy")

    def _backward():
        # dL/dz = (probs - one_hot(targets)) [/ N if reduction='mean']
        grad = probs.copy()
        flat_grad = grad.reshape(-1, grad.shape[-1])
        flat_grad[xp.arange(N), flat_targets] -= 1.0
        if reduction == "mean":
            grad /= N
        logits.grad += grad * out.grad

    out._backward = _backward
    return out
