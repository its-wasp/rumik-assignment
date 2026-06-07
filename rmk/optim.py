"""AdamW optimizer, cosine LR schedule with warmup, global-norm gradient clipping."""

from rmk.backend import xp


class AdamW:
    """AdamW with decoupled weight decay. weight_decay=0 reduces to plain Adam."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.b1, self.b2 = betas
        self.state = {}  # id(p) -> {"m": ndarray, "v": ndarray}
        self.t = 0

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p in self.params:
            g = p.grad
            st = self.state.setdefault(
                id(p),
                {"m": xp.zeros_like(p.data), "v": xp.zeros_like(p.data)},
            )
            # inplace EMA updates of the first and second moments
            st["m"] *= self.b1
            st["m"] += (1.0 - self.b1) * g
            st["v"] *= self.b2
            st["v"] += (1.0 - self.b2) * (g * g)
            m_hat = st["m"] / bc1
            v_hat = st["v"] / bc2
            if self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


def cosine_with_warmup(step, warmup, max_steps, max_lr, min_lr=0.0):
    """Linear ramp 0 -> max_lr over `warmup`; then half-cosine decay to `min_lr` by `max_steps`."""
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + float(xp.cos(xp.pi * progress)))


def clip_grad_norm(params, max_norm):
    """Global L2 clip: scale all grads down if their joint L2 norm exceeds `max_norm`.

    """
    sq = 0.0
    for p in params:
        sq += float((p.grad * p.grad).sum())
    norm = sq ** 0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for p in params:
            p.grad *= scale
    return norm
