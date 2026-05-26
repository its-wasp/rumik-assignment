"""
Tensor with reverse-mode autograd over xp.ndarrays.

Each Tensor node holds an ndarray value (`data`), an accumulated gradient
(`grad`, same shape as data), a tuple of parent Tensors (`_parents`) that
produced it, and a closure (`_backward`) that knows how to push gradients
from this node back to its parents using the chain rule.

`backward()` builds a topological order of the graph reachable from `self`,
seeds `self.grad = xp.ones_like(self.data)`, and walks the order in reverse
so every node's gradient is fully accumulated by the time its `_backward`
fires.

Scalars are represented as 0-d ndarrays so the code path is uniform.
"""

from rmk.backend import xp


def _unbroadcast(grad, shape):
    """Sum `grad` along axes that were broadcast to produce its current shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _parents=(), _op=""):
        self.data = xp.asarray(data, dtype=xp.float64)
        self.grad = xp.zeros_like(self.data)
        self._parents = _parents
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,), "neg")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    # right-hand variants so `2 + a`, `2 * a`, `2 - a` work
    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return Tensor(other) - self

    def exp(self):
        out = Tensor(xp.exp(self.data), (self,), "exp")

        def _backward():
            # d(exp(x))/dx = exp(x) = out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(xp.log(self.data), (self,), "log")

        def _backward():
            # d(log(x))/dx = 1/x
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def __pow__(self, p):
        if isinstance(p, Tensor):
            raise TypeError("Tensor ** Tensor not supported; exponent must be a number.")
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            # d(x**p)/dx = p * x**(p-1)
            self.grad += p * self.data ** (p - 1) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                for ax in sorted(a if a >= 0 else a + self.data.ndim for a in axes):
                    grad = xp.expand_dims(grad, ax)
            self.grad += xp.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ xp.swapaxes(other.data, -1, -2)
            other.grad += xp.swapaxes(self.data, -1, -2) @ out.grad

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(self.data.transpose(axes), (self,), "T")

        def _backward():
            if axes is None:
                self.grad += out.grad.transpose()
            else:
                inv = [0] * len(axes)
                for i, a in enumerate(axes):
                    inv[a] = i
                self.grad += out.grad.transpose(inv)

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def zero_grad(self):
        """Reset .grad on this node and every ancestor reachable through parents."""
        seen = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            node.grad = xp.zeros_like(node.data)
            stack.extend(node._parents)

    def backward(self):
        """Reverse-mode autograd: topo-sort the graph, seed, walk in reverse."""
        topo = []
        seen = set()

        def visit(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._parents:
                visit(p)
            topo.append(v)

        visit(self)
        self.grad = xp.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
