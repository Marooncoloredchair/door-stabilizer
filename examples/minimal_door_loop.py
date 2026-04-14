"""Minimal Door loop: drifting quadratic–style toy (numpy only)."""

from __future__ import annotations

import numpy as np

from door import Door


class QuadraticPlant:
    """maximize -||a - target||^2 with random-walk target (toy drift)."""

    def __init__(self, dim: int, seed: int = 0) -> None:
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.target = self.rng.standard_normal(dim)

    def step(self, a: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).ravel()
        err = a - self.target
        r = float(-np.dot(err, err))
        self.target += 0.01 * self.rng.standard_normal(self.dim)
        return r


def main() -> None:
    dim = 2
    plant = QuadraticPlant(dim=dim, seed=1)
    ctrl = Door(dim=dim, action_low=-1.0, action_high=1.0, seed=0)
    total = 0.0
    for t in range(200):
        u = ctrl.act()
        r = plant.step(u)
        ctrl.update(r)
        total += r
        if t % 50 == 0:
            print(f"t={t:4d}  step_reward={r:8.4f}  cumulative={total:10.2f}")
    print("done cumulative reward:", total)


if __name__ == "__main__":
    main()
