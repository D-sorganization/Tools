"""Regression tests for calc backend ODE solver preconditions."""

from __future__ import annotations

import pytest
from calc_backend.routers.ode_solver import _rk4_solve


def test_rk4_solve_rejects_single_output_point() -> None:
    """Direct RK4 calls should enforce the public request point-count contract."""
    with pytest.raises(ValueError, match="num_points must be at least 2"):
        _rk4_solve(
            var_names=["y"],
            expressions={"y": "-y"},
            parameters={},
            initial={"y": 1.0},
            t_start=0.0,
            t_end=1.0,
            num_points=1,
        )
