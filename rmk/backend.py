import os

_BACKEND = os.environ.get("RMK_BACKEND", "numpy").lower()

if _BACKEND == "numpy":
    import numpy as xp
elif _BACKEND == "cupy":
    import cupy as xp
else:
    raise ValueError(f"Unknown RMK_BACKEND: {_BACKEND!r} (expected 'numpy' or 'cupy')")

name = _BACKEND
