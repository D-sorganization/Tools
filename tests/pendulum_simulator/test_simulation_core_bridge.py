"""Bridge test ensuring integrate_ode is tested through top-level tests.

Specifically verifies integrate_ode in tests/pendulum_simulator.
"""

from __future__ import annotations

import numpy as np
from double_pendulum_golf.simulation_core import integrate_ode


def test_integrate_ode_top_level_coverage() -> None:
    """Ensure integrate_ode executes and includes t_end in top-level test runner."""
    t_end = 1.0
    dt = 0.1
    t, states = integrate_ode(lambda t, y: -y, np.array([1.0]), t_end=t_end, dt=dt)
    assert len(t) == 11
    assert np.isclose(t[-1], t_end, atol=1e-12)
    assert np.isclose(t[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(states[-1, 0], np.exp(-1.0), rtol=1e-4)
