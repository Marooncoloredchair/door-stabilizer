"""
Plot cumulative reward: Door vs random actions on the same toy plant.

Requires: pip install door-stabilizer matplotlib

This is a *toy* demo, not a claim about real PID loops or hardware.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Install matplotlib: pip install matplotlib") from e

from door import Door


def run_door(steps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    target = rng.standard_normal(2)
    ctrl = Door(dim=2, action_low=-1.0, action_high=1.0, seed=seed)
    cum = 0.0
    prefix: list[float] = []
    for _ in range(steps):
        u = ctrl.act()
        err = u - target
        r = float(-np.dot(err, err))
        target += 0.01 * rng.standard_normal(2)
        ctrl.update(r)
        cum += r
        prefix.append(cum)
    return np.asarray(prefix, dtype=np.float64)


def run_random(steps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    target = rng.standard_normal(2)
    act_rng = np.random.default_rng(seed + 99)
    cum = 0.0
    prefix: list[float] = []
    for _ in range(steps):
        u = act_rng.uniform(-1.0, 1.0, size=2)
        err = u - target
        r = float(-np.dot(err, err))
        target += 0.01 * rng.standard_normal(2)
        cum += r
        prefix.append(cum)
    return np.asarray(prefix, dtype=np.float64)


def main() -> None:
    steps = 200
    seed = 0
    y_door = run_door(steps, seed)
    y_rand = run_random(steps, seed)
    t = np.arange(1, steps + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(t, y_door, label="Door (online)", color="#2563eb", linewidth=2)
    plt.plot(t, y_rand, label="Random actions (no learning)", color="#94a3b8", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Cumulative reward")
    plt.title("Toy drifting quadratic: higher is better")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "door_vs_random_cumulative.png"
    plt.savefig(out, dpi=150)
    print("Wrote", out)


if __name__ == "__main__":
    main()
