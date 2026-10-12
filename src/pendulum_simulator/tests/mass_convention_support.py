"""Shared constants and helpers for the test_mass_convention* modules (Tools #5431)."""

from __future__ import annotations

import pathlib

import numpy as np

from double_pendulum_golf.physics import PendulumParams

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "distal_inertia_cases.json"
REPRODUCER = PendulumParams(m1=2.0, m2=3.0, mClub=1.0, L1=0.7, L2=1.0)
GOLF = PendulumParams(m1=5.0, m2=0.108, mClub=0.202, L1=0.65, L2=1.10, b1=0.1, b2=0.05)


def random_states(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, size=(count, 4))


def params_cases() -> list[PendulumParams]:
    rng = np.random.default_rng(7)
    return [
        PendulumParams(
            m1=float(rng.uniform(1, 6)),
            m2=float(rng.uniform(0.05, 3)),
            mClub=float(rng.uniform(0, 1.5)),
            L1=float(rng.uniform(0.3, 1)),
            L2=float(rng.uniform(0.5, 1.5)),
        )
        for _ in range(12)
    ]
