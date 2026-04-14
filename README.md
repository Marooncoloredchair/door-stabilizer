# Door (door-stabilizer)

[![PyPI version](https://img.shields.io/pypi/v/door-stabilizer.svg)](https://pypi.org/project/door-stabilizer/)
[![Python](https://img.shields.io/pypi/pyversions/door-stabilizer.svg)](https://pypi.org/project/door-stabilizer/)
[![PyPI - License](https://img.shields.io/pypi/l/door-stabilizer)](https://pypi.org/project/door-stabilizer/)

**Door** is a small **online adaptive control** library for Python. You implement a **plant** `step(action) → reward`, then loop with **`Door.act()`** and **`Door.update(reward)`**. Internally it maintains an **online surrogate** (ridge by default; optional **PyTorch** MLP), proposes and refines actions, and widens exploration when reward variability spikes. A batch API **`door.run`** is available with optional **N(t)**-style stability readouts.

**Install from PyPI** (import name is still `door`):

```bash
pip install door-stabilizer
```

Optional PyTorch surrogate:

```bash
pip install "door-stabilizer[torch]"
```

---

## Why use this?

- **Minimal API**: one incremental loop — no framework lock-in.
- **Same-physics budget** baselines: compare against CEM-restart style search on the same `plant.step` call budget (see benchmarks in the upstream research tree).
- **Lightweight core**: NumPy-only path works without PyTorch.

---

## Minimal example

```python
import numpy as np
from door import Door

class Plant:
    def step(self, a):
        return float(-np.sum(np.asarray(a) ** 2))

p = Plant()
ctrl = Door(dim=2, action_low=-1.0, action_high=1.0, seed=0)
for _ in range(100):
    u = ctrl.act()
    r = p.step(u)
    ctrl.update(r)
```

More examples live in [`examples/`](examples/).

---

## API sketch

| Piece | Role |
|--------|------|
| `Door.act()` | Propose next action from the current surrogate + exploration. |
| `Door.update(reward)` | Ingest reward and update the online model. |
| `door.run` | Batch rollout API with optional volatility widening and **N(t)** summary helpers. |

Aliases: `HAT` is kept as an alias for `Door` for compatibility with older notebooks.

---

## Version

Current release line: **0.4.x** on [PyPI — door-stabilizer](https://pypi.org/project/door-stabilizer/).

```bash
python -c "import door; print(door.__version__)"
```

---

## Repository name on GitHub

This GitHub project may appear as **`door-stabalizer`** (typo). The **PyPI distribution name** is **`door-stabilizer`**. Consider renaming the GitHub repository to **`door-stabilizer`** so `pip` and docs stay aligned.

---

## License

See [PyPI package metadata](https://pypi.org/project/door-stabilizer/) (Proprietary — BioQuant).

---

## Issues

Use **GitHub Issues** on this repository for install problems or documentation fixes.
