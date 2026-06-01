"""Transformer building blocks: MLP, MultiHeadAttention (causal), Block."""

from rmk.backend import xp
from rmk.tensor import Tensor
from rmk.functional import gelu, softmax
from rmk.nn import Module, Linear, LayerNorm, Dropout


class MLP(Module):
    """Position-wise feed-forward: Linear -> GELU -> Linear -> Dropout. Hidden = 4 * n_embd."""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.fc1 = Linear(n_embd, 4 * n_embd)
        self.fc2 = Linear(4 * n_embd, n_embd)
        self.drop = Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(gelu(self.fc1(x))))


class MultiHeadAttention(Module):
    """Causal multi-head self-attention. 3 separate Q/K/V projections + output proj."""

    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = Linear(n_embd, n_embd)
        self.k_proj = Linear(n_embd, n_embd)
        self.v_proj = Linear(n_embd, n_embd)
        self.out_proj = Linear(n_embd, n_embd)
        self.attn_drop = Dropout(dropout)
        self.resid_drop = Dropout(dropout)

    def forward(self, x):
        B, T, C = x.data.shape
        nh, hd = self.n_head, self.head_dim

        def split(t):
            # (B, T, C) -> (B, T, nh, hd) -> (B, nh, T, hd)
            return t.reshape(B, T, nh, hd).transpose((0, 2, 1, 3))

        q = split(self.q_proj(x))
        k = split(self.k_proj(x))
        v = split(self.v_proj(x))

        # scaled dot-product scores: (B, nh, T, T)
        scale = 1.0 / xp.sqrt(hd)
        scores = (q @ k.transpose((0, 1, 3, 2))) * scale

        # causal mask: -1e9 above the diagonal; broadcasts across (B, nh)
        mask = xp.triu(xp.ones((T, T)) * -1e9, k=1)
        scores = scores + mask

        weights = self.attn_drop(softmax(scores, axis=-1))
        out = weights @ v  # (B, nh, T, hd)

        # merge heads: (B, nh, T, hd) -> (B, T, nh, hd) -> (B, T, C)
        out = out.transpose((0, 2, 1, 3)).reshape(B, T, C)
        return self.resid_drop(self.out_proj(out))


class Block(Module):
    """Pre-LN transformer block: x + attn(LN(x)); x + mlp(LN(x))."""

    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
