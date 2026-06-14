import os

_BACKEND = os.environ.get("RMK_BACKEND", "numpy").lower()

if _BACKEND == "numpy":
    import numpy as xp
elif _BACKEND == "cupy":
    import cupy as xp
else:
    raise ValueError(f"Unknown RMK_BACKEND: {_BACKEND!r} (expected 'numpy' or 'cupy')")

name = _BACKEND

# Default Tensor dtype. Read at Tensor construction time, so training scripts can
# override this BEFORE building the model (e.g. set DTYPE = xp.float32 for GPU runs).
# Tests don't touch it -> stays fp64 -> gradcheck tolerance unchanged.
DTYPE = xp.float64
