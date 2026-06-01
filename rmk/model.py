"""GPT decoder-only transformer with weight-tied LM head."""

from dataclasses import dataclass

from rmk.backend import xp
from rmk.losses import cross_entropy
from rmk.nn import Module, Embedding, LayerNorm, Dropout, ModuleList
from rmk.transformer import Block


@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.0


class GPT(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.drop = Dropout(config.dropout)
        self.blocks = ModuleList([
            Block(config.n_embd, config.n_head, config.dropout)
            for _ in range(config.n_layer)
        ])
        self.ln_f = LayerNorm(config.n_embd)
        self._apply_scaled_init()

    def _apply_scaled_init(self):
        # residual stream variance bounded as depth grows by scaling.
        std = 0.02 / xp.sqrt(2 * self.config.n_layer)
        for block in self.blocks:
            for w in (block.attn.out_proj.weight, block.mlp.fc2.weight):
                w.data[...] = xp.random.standard_normal(w.data.shape) * std

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        pos = xp.arange(T)
        tok_emb = self.wte(idx)        # (B, T, C)
        pos_emb = self.wpe(pos)        # (T, C) -- broadcasts in the add
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # weight tying: LM head shares wte.weight (no separate lm_head module)
        logits = x @ self.wte.weight.transpose()   # (B, T, V)
        loss = cross_entropy(logits, targets) if targets is not None else None
        return logits, loss
