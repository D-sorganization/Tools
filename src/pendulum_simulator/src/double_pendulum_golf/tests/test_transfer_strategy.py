"""Contract tests for phase-resolved drift-transfer strategy metrics."""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.transfer_strategy import (
    TransferSignals,
    double_pendulum_transfer_signals,
    pareto_front,
    summarize_transfer,
)
from double_pendulum_golf.physics import PendulumParams
from double_pendulum_golf.simulation import SimulationResult


def _signals() -> TransferSignals:
    time = np.array([0.0, 0.5, 1.0])
    grip_velocity = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    force_drift = np.array([[4.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
    force_control = np.array([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    return TransferSignals(
        time_s=time,
        proximal_angular_velocity_rad_s=np.array([5.0, 6.0, 7.0]),
        distal_speed_m_s=np.array([10.0, 14.0, 18.0]),
        distal_kinetic_energy_j=np.array([8.0, 14.0, 22.0]),
        grip_velocity_m_s=grip_velocity,
        grip_force_total_n=force_drift + force_control,
        grip_force_drift_n=force_drift,
        grip_force_control_n=force_control,
        wrist_control_couple_nm=np.array([-2.0, -2.0, -2.0]),
        club_angular_velocity_rad_s=np.array([3.0, 4.0, 5.0]),
        model_tier="test_double_pendulum",
    )


def test_summary_closes_work_and_reports_braking() -> None:
    summary = summarize_transfer(_signals(), start_s=0.0, end_s=1.0)

    assert summary.total_grip_work_j == pytest.approx(0.5)
    assert summary.drift_grip_work_j == pytest.approx(1.5)
    assert summary.control_grip_work_j == pytest.approx(-1.0)
    assert summary.work_closure_residual_j == pytest.approx(0.0)
    assert summary.negative_grip_work_j == pytest.approx(-0.5625)
    assert summary.negative_along_path_impulse_n_s == pytest.approx(-0.5625)
    assert summary.wrist_control_work_j == pytest.approx(-8.0)
    assert summary.distal_energy_gain_j == pytest.approx(14.0)
    assert summary.peak_distal_speed_m_s == pytest.approx(18.0)


def test_summary_rejects_undefined_or_out_of_range_window() -> None:
    signals = _signals()
    with pytest.raises(ValueError, match="start_s must be less"):
        summarize_transfer(signals, start_s=0.5, end_s=0.5)
    with pytest.raises(ValueError, match="inside the trajectory"):
        summarize_transfer(signals, start_s=-0.1, end_s=0.5)


def test_signals_enforce_exact_drift_control_closure() -> None:
    signals = _signals()
    with pytest.raises(ValueError, match="closure"):
        TransferSignals(
            time_s=signals.time_s,
            proximal_angular_velocity_rad_s=signals.proximal_angular_velocity_rad_s,
            distal_speed_m_s=signals.distal_speed_m_s,
            distal_kinetic_energy_j=signals.distal_kinetic_energy_j,
            grip_velocity_m_s=signals.grip_velocity_m_s,
            grip_force_total_n=signals.grip_force_total_n + 0.1,
            grip_force_drift_n=signals.grip_force_drift_n,
            grip_force_control_n=signals.grip_force_control_n,
            wrist_control_couple_nm=signals.wrist_control_couple_nm,
            club_angular_velocity_rad_s=signals.club_angular_velocity_rad_s,
            model_tier="broken",
        )


def test_pareto_front_preserves_tradeoffs_and_rejects_dominated_rows() -> None:
    rows = np.array(
        [
            [18.0, 1.0, 5.0],
            [17.0, 0.5, 4.0],
            [16.0, 2.0, 6.0],
        ]
    )

    indices = pareto_front(rows, maximize=(True, False, False))

    assert np.array_equal(indices, np.array([0, 1]))


def test_pareto_front_validates_objective_contract() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        pareto_front(np.array([1.0, 2.0]), maximize=(True,))
    with pytest.raises(ValueError, match="maximize length"):
        pareto_front(np.ones((2, 2)), maximize=(True,))


def test_double_adapter_preserves_coordinate_meaning_and_force_closure() -> None:
    params = PendulumParams(m1=4.0, m2=0.3, L1=0.7, L2=1.0, mClub=0.2)
    result = SimulationResult(
        t=np.array([0.0, 0.1, 0.2]),
        states=np.array(
            [
                [-0.8, -1.1, 4.0, 0.5],
                [-0.35, -0.9, 5.0, 1.5],
                [0.2, -0.55, 6.0, 3.0],
            ]
        ),
        params=params,
        torque_func=lambda _time: (30.0, -4.0),
    )

    signals = double_pendulum_transfer_signals(result)

    assert signals.model_tier == "exact_planar_double_pendulum"
    assert np.array_equal(signals.proximal_angular_velocity_rad_s, result.dtheta1)
    assert np.allclose(
        signals.grip_force_total_n,
        signals.grip_force_drift_n + signals.grip_force_control_n,
    )
    assert np.all(signals.distal_kinetic_energy_j > 0.0)
    assert np.array_equal(signals.wrist_control_couple_nm, np.full(3, -4.0))
