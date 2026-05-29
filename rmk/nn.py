"""Module base class + basic layers. Auto-registers Tensors as parameters."""

from rmk.backend import xp
from rmk.tensor import Tensor
from rmk.functional import embedding, layer_norm


class Module:
    def __init__(self):
        # bootstrap via object.__setattr__ so the next __setattr__ calls can find these dicts
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        # dedup by id() so weight-tied tensors aren't stepped on twice relevant to LLMs initial embedding layer and 
        # at last vocab projection (transposing the same embedding layer weights) -> shared memory
        seen, out = set(), []

        def walk(m):
            for p in m._params.values():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
            for child in m._modules.values():
                walk(child)

        walk(self)
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self, prefix=""):
        # p.data.copy() because if we don't do that we are going to just update the references.
        sd = {prefix + n: p.data.copy() for n, p in self._params.items()}
        for n, m in self._modules.items():
            sd.update(m.state_dict(prefix + n + "."))
        return sd

    def load_state_dict(self, sd, prefix=""):
        for n, p in self._params.items():
            key = prefix + n
            if key in sd:
                # changing the values inplace without changing the pointer.
                p.data[...] = sd[key]
        for n, m in self._modules.items():
            m.load_state_dict(sd, prefix + n + ".")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Tensor(xp.random.standard_normal((in_features, out_features)) * 0.02)
        if bias:
            self.bias = Tensor(xp.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Tensor(xp.random.standard_normal((num_embeddings, embedding_dim)) * 0.02)

    def forward(self, ids):
        return embedding(self.weight, ids)


class LayerNorm(Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.weight = Tensor(xp.ones(features))
        self.bias = Tensor(xp.zeros(features))
        self.eps = eps

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.eps)


class Dropout(Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    # inverted dropout method
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = (xp.random.random(x.data.shape) >= self.p).astype(x.data.dtype)
        scale = 1.0 / (1.0 - self.p)
        out = Tensor(x.data * mask * scale, (x,), "dropout")

        def _backward():
            x.grad += out.grad * mask * scale

        out._backward = _backward
        return out
