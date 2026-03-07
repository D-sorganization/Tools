#!/usr/bin/env python3
"""Measure simulation performance."""

import time
import numpy as np
from src.double_pendulum_golf.physics import PendulumParams
from src.double_pendulum_golf.simulation import make_polynomial_torque, run_simulation

# Default preset params
params = PendulumParams(
    m1=5.0,
    m2=0.5,
    L1=0.6,
    L2=1.0,
)

initial_state = np.array(
    [
        np.radians(120.0),
        np.radians(-90.0),
        0.0,
        0.0,
    ]
)

torque_func = make_polynomial_torque([-25, 10], [0])

print("Running simulation 10 times to measure performance...")  # noqa: T201
times = []
for i in range(10):
    start = time.time()
    result = run_simulation(
        params=params,
        initial_state=initial_state,
        t_end=2.0,
        torque_func=torque_func,
        dt=0.005,
    )
    elapsed = time.time() - start
    times.append(elapsed)
<<<<<<< Updated upstream
    print(f"  Run {i+1}: {elapsed:.3f}s")  # noqa: T201
=======
    print(f"  Run {i + 1}: {elapsed:.3f}s")
>>>>>>> Stashed changes

print(f"\nAverage: {np.mean(times):.3f}s")  # noqa: T201
print(f"Min: {np.min(times):.3f}s")  # noqa: T201
print(f"Max: {np.max(times):.3f}s")  # noqa: T201
