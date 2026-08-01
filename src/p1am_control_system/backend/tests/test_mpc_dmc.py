"""Numerical tests for the Dynamic Matrix Control move solver (mpc.py).

DMC is an *incremental* formulation: the free response already contains the
full predicted effect of holding the current control value, so the decision
variable is the move vector ``delta_u``. These tests pin that contract -- most
importantly that a process already sitting on setpoint is asked to move zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from mpc import CV_MAX, MAX_MOVE, simulate_pid_vs_mpc, solve_dmc_move

# Nominal FOPDT plant shared by the move-solver tests.
PLANT = {
    "process_gain": 1.2,
    "process_tau": 5.0,
    "process_delay": 1.0,
    "dt": 0.5,
    "prediction_horizon": 10,
    "control_horizon": 3,
    "rho": 0.1,
}


@dataclass
class _Payload:
    """Stand-in for the FastAPI ``MPCSimulatePayload`` pydantic model."""

    prediction_horizon: int = 10
    control_horizon: int = 3
    setpoint: float = 50.0
    rho: float = 0.1
    process_gain: float = 1.2
    process_tau: float = 5.0
    process_delay: float = 1.0


@pytest.mark.unit
@pytest.mark.parametrize("setpoint", [20.0, 50.0, 80.0])
def test_steady_state_move_is_zero(setpoint: float) -> None:
    """At steady state on setpoint the optimal move is exactly zero.

    With ``pv == setpoint`` and the control value already holding the process
    there (``last_cv == setpoint / Kp``), the free response equals the target
    over the whole horizon, so the DMC objective is minimised by ``delta_u=0``.
    Any nonzero move means the current input is being counted twice.
    """
    kp = float(PLANT["process_gain"])
    move = solve_dmc_move(
        pv=setpoint,
        last_cv=setpoint / kp,
        setpoint=setpoint,
        **PLANT,
    )
    assert move == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_move_sign_follows_error() -> None:
    """The first move pushes the CV toward the setpoint, not away from it."""
    kp = float(PLANT["process_gain"])
    setpoint = 50.0
    hold_cv = setpoint / kp

    below = solve_dmc_move(
        pv=setpoint - 10.0,
        last_cv=hold_cv,
        setpoint=setpoint,
        **PLANT,
    )
    above = solve_dmc_move(
        pv=setpoint + 10.0,
        last_cv=hold_cv,
        setpoint=setpoint,
        **PLANT,
    )
    assert below > 0.0
    assert above < 0.0


@pytest.mark.unit
def test_move_is_bounded() -> None:
    """The solver bounds the move, not the absolute CV."""
    move = solve_dmc_move(
        pv=0.0,
        last_cv=0.0,
        setpoint=100.0,
        **PLANT,
    )
    assert abs(move) <= MAX_MOVE


@pytest.mark.unit
@pytest.mark.parametrize("setpoint", [20.0, 50.0])
def test_simulation_has_no_steady_state_offset(setpoint: float) -> None:
    """The MPC trace settles on setpoint instead of parking below it.

    Counting the current input twice made the closed loop settle at roughly
    half the required CV, so the comparison chart showed MPC with a permanent
    offset at any nonzero operating point.
    """
    result = simulate_pid_vs_mpc(_Payload(setpoint=setpoint))
    final_pv = result["mpc"]["pv"][-1]
    final_cv = result["mpc"]["cv"][-1]
    assert final_pv == pytest.approx(setpoint, rel=0.05)
    assert final_cv == pytest.approx(setpoint / 1.2, rel=0.05)


@pytest.mark.unit
def test_simulation_respects_cv_limits() -> None:
    """Integrated moves never drive the CV outside the 0-100% output range."""
    result = simulate_pid_vs_mpc(_Payload(setpoint=100.0, process_gain=0.5))
    assert all(0.0 <= cv <= CV_MAX for cv in result["mpc"]["cv"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pv", "50"),
        ("last_cv", None),
        ("setpoint", [50.0]),
        ("process_gain", "1.2"),
        ("prediction_horizon", 10.0),
        ("control_horizon", "3"),
    ],
)
def test_solve_dmc_move_rejects_wrong_types(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "pv": 0.0,
        "last_cv": 0.0,
        "setpoint": 50.0,
        **PLANT,
    }
    kwargs[field] = value
    with pytest.raises(TypeError):
        solve_dmc_move(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("process_tau", 0.0),
        ("process_tau", -1.0),
        ("process_delay", -0.5),
        ("dt", 0.0),
        ("rho", -0.1),
        ("prediction_horizon", 0),
        ("control_horizon", 0),
        ("pv", float("nan")),
        ("setpoint", float("inf")),
    ],
)
def test_solve_dmc_move_rejects_out_of_range(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "pv": 0.0,
        "last_cv": 0.0,
        "setpoint": 50.0,
        **PLANT,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        solve_dmc_move(**kwargs)


@pytest.mark.unit
def test_solve_dmc_move_rejects_control_horizon_above_prediction() -> None:
    kwargs: dict[str, object] = {
        "pv": 0.0,
        "last_cv": 0.0,
        "setpoint": 50.0,
        **PLANT,
    }
    kwargs["control_horizon"] = 11
    with pytest.raises(ValueError):
        solve_dmc_move(**kwargs)
