# rmk

A from-scratch implementation of a GPT-2-style decoder-only transformer **without using torch or jax autograd**. The autograd engine, every layer, and every gradient are implemented by hand on top of NumPy (and CuPy, for GPU training).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
## More coming soon..
