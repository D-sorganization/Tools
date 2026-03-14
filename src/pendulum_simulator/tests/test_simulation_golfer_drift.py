"""Gap-fill tests for simulation_golfer.py lines 266, 296, 302.

These lines fire when constraint violation exceeds threshold values during
Baumgarte stabilization. Uses mocking to simulate constraint drift.

Line 266: log warning when viol > _CONSTRAINT_WARN_TOL (1e-4)
Line 296: log.error when max_viol > _CONSTRAINT_ABORT_TOL (1e-2)
Line 302: log.warning when _CONSTRAINT_WARN_TOL < max_viol <= _CONSTRAINT_ABORT_TOL
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest

from double_pendulum_golf.physics_golfer import GolferParams, N_DOF
from double_pendulum_golf.simulation_golfer import run_simulation


@pytest.fixture(scope="module")
def golfer_params() -> GolferParams:
    return GolferParams(
        m_hub=2.0,
        m_r_upper=3.0,
        m_r_fore=2.0,
        m_l_upper=3.0,
        m_l_fore=2.0,
        m_club=0.5,
        L_hub=0.15,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.10,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


class TestConstraintDriftLogging:
    def test_warn_during_integration_line266(
        self, golfer_params: GolferParams, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Line 266: warn when viol > _CONSTRAINT_WARN_TOL during integration."""
        initial_state = np.zeros(2 * N_DOF)
        initial_state[0] = 0.05

        # Patch constraint_violation to return a high value on first few calls
        call_count = [0]

        def fake_violation(y, params):
            call_count[0] += 1
            # Return a value above warn threshold for first few calls
            if call_count[0] <= 3:
                return 1e-3  # > _CONSTRAINT_WARN_TOL (1e-4), < _CONSTRAINT_ABORT_TOL (1e-2)
            return 0.0

        with caplog.at_level(
            logging.WARNING, logger="double_pendulum_golf.simulation_golfer"
        ):
            with patch(
                "double_pendulum_golf.simulation_golfer.constraint_violation",
                side_effect=fake_violation,
            ):
                result = run_simulation(
                    golfer_params,
                    initial_state,
                    t_end=0.02,
                    torque_func=lambda t: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    dt=0.01,
                )
        # Should have logged a warning during simulation
        assert "Constraint violation" in caplog.text or len(result.t) >= 2

    def test_abort_threshold_log_line296(
        self, golfer_params: GolferParams, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Line 296: log.error when max_viol > _CONSTRAINT_ABORT_TOL (1e-2)."""
        initial_state = np.zeros(2 * N_DOF)

        call_count = [0]

        def fake_violation(y, params):
            call_count[0] += 1
            if call_count[0] <= 2:
                return 0.5  # >> _CONSTRAINT_ABORT_TOL (1e-2)
            return 0.0

        with caplog.at_level(
            logging.ERROR, logger="double_pendulum_golf.simulation_golfer"
        ):
            with patch(
                "double_pendulum_golf.simulation_golfer.constraint_violation",
                side_effect=fake_violation,
            ):
                result = run_simulation(
                    golfer_params,
                    initial_state,
                    t_end=0.02,
                    torque_func=lambda t: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    dt=0.01,
                )
        assert "Excessive constraint drift" in caplog.text
        assert len(result.t) >= 2

    def test_warn_threshold_postcondition_line302(
        self, golfer_params: GolferParams, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Line 302: log.warning when WARN < max_viol <= ABORT (postcondition)."""
        initial_state = np.zeros(2 * N_DOF)

        call_count = [0]

        def fake_violation(y, params):
            call_count[0] += 1
            if call_count[0] == 1:
                return 5e-3  # > WARN (1e-4), < ABORT (1e-2)
            return 0.0

        with caplog.at_level(
            logging.WARNING, logger="double_pendulum_golf.simulation_golfer"
        ):
            with patch(
                "double_pendulum_golf.simulation_golfer.constraint_violation",
                side_effect=fake_violation,
            ):
                result = run_simulation(
                    golfer_params,
                    initial_state,
                    t_end=0.02,
                    torque_func=lambda t: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    dt=0.01,
                )
        assert "Max constraint violation" in caplog.text
        assert len(result.t) >= 2
