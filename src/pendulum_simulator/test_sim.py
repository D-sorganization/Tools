#!/usr/bin/env python3
"""Quick test to see if simulation runs without GUI."""

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
        np.radians(120.0),  # theta1
        np.radians(-90.0),  # phi
        0.0,  # dtheta1
        0.0,  # dphi
    ]
)

torque_func = make_polynomial_torque(
    [-25, 10],  # shoulder
    [0],  # wrist
)

print("Running simulation...")
try:
    result = run_simulation(
        params=params,
        initial_state=initial_state,
        t_end=2.0,
        torque_func=torque_func,
        dt=0.005,
    )
    print("✓ Simulation succeeded!")
    print(f"  Steps: {result.n_steps}")
    print(f"  Time range: {result.t[0]:.3f} to {result.t[-1]:.3f} s")
    print(f"  Initial state: {result.states[0]}")
    print(f"  Final state: {result.states[-1]}")
except Exception as e:  # noqa: BLE001
    print(f"✗ Simulation failed: {e}")
    import traceback

    traceback.print_exc()
