"""Contract tests for the slew-limited downswing collocation optimizer.

Two of these tests exist because the naive configuration silently produces
answers that look successful and are not:

* ``test_loose_tolerance_regression_pin`` — at SciPy's default ``ftol`` the
  solver reports success after a handful of iterations having done nothing but
  find a feasible trajectory, and hands back the initial guess.
* ``test_slew_limit_is_binding_and_changes_the_answer`` — without a torque
  slew-rate limit the optimizer buys clubhead speed with instantaneous
  full-torque reversals no golfer could produce.

Closes #4769.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import JointLimits, PendulumParams, TorqueClamp
from double_pendulum_golf.swing_objectives.downswing import (
    DownswingConfig,
    DownswingOptimizer,
    DownswingResult,
)

_PARAMS = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10, mClub=0.20)

_TORQUE_CLAMP = TorqueClamp(max_torque1=180.0, max_torque2=20.0)
_WRIST_LIMITS = JointLimits(phi_min=-0.175, phi_max=2.094, theta1_min=-4.0, theta1_max=4.0)

_FEASIBLE_DEFECT = 1e-9


def _config(**overrides: object) -> DownswingConfig:
    """Build a downswing configuration with tour-plausible defaults."""
    defaults: dict[str, object] = {
        "params": _PARAMS,
        "node_count": 21,
        "duration_s": 0.34,
        "initial_state": np.array([2.618, 1.745, 0.0, 0.0]),
        "impact_theta1_rad": 0.0,
        "torque_clamp": _TORQUE_CLAMP,
        "joint_limits": _WRIST_LIMITS,
    }
    defaults.update(overrides)
    return DownswingConfig(**defaults)  # type: ignore[arg-type]


def test_config_rejects_a_downswing_the_golfer_cannot_deliver() -> None:
    """A duration below the torque-budget bound fails with an actionable message.

    Without this screen the solver reports "positive directional derivative for
    linesearch", which says nothing about the golfer. The bound is necessary,
    not sufficient — the slew ramp and the wrist release push the true minimum
    higher — and the message says so.
    """
    config = _config()
    assert config.minimum_sweep_duration_s == pytest.approx(0.2856, abs=1e-3)

    with pytest.raises(ValueError, match="below the"):
        _config(duration_s=0.20)


def test_config_rejects_degenerate_settings() -> None:
    """Contract: an unusable configuration fails at construction, not at solve."""
    with pytest.raises(ValueError, match="node_count"):
        _config(node_count=3)
    with pytest.raises(ValueError, match="duration_s"):
        _config(duration_s=0.0)
    with pytest.raises(ValueError, match="initial_state"):
        _config(initial_state=np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        _config(initial_state=np.array([np.nan, 0.0, 0.0, 0.0]))


def test_config_start_must_respect_the_wrist_limits() -> None:
    """A top-of-backswing posture outside the anatomical range is rejected."""
    with pytest.raises(ValueError, match="wrist"):
        _config(initial_state=np.array([2.618, 3.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    "objective_key",
    ["clubhead_speed", "centrifugal", "coriolis", "energy_transfer", "impulse_transfer"],
)
def test_every_objective_converges_to_a_feasible_swing(objective_key: str) -> None:
    """Each objective must return a dynamically feasible, legal downswing."""
    result = DownswingOptimizer(_config()).solve(objective_key)

    assert isinstance(result, DownswingResult)
    assert result.success, f"{objective_key} failed: {result.message}"
    assert result.feasible, f"{objective_key} defect {result.max_defect:.2e}"
    assert result.max_defect < _FEASIBLE_DEFECT
    assert np.isfinite(result.objective_value)
    assert result.objective.key == objective_key


def test_solution_satisfies_the_boundary_conditions() -> None:
    """The swing starts at the top and arrives at the ball."""
    result = DownswingOptimizer(_config()).solve("clubhead_speed")
    states = result.states

    assert np.allclose(states[0], _config().initial_state, atol=1e-6)
    assert states[-1, 0] == pytest.approx(0.0, abs=1e-6)


def test_solution_respects_torque_and_wrist_limits() -> None:
    """Bounds are hard: no sample may exceed the golfer's declared capability."""
    result = DownswingOptimizer(_config()).solve("clubhead_speed")

    assert np.all(np.abs(result.torques[:, 0]) <= _TORQUE_CLAMP.max_torque1 + 1e-6)
    assert np.all(np.abs(result.torques[:, 1]) <= _TORQUE_CLAMP.max_torque2 + 1e-6)
    assert result.states[:, 1].min() >= _WRIST_LIMITS.phi_min - 1e-6
    assert result.states[:, 1].max() <= _WRIST_LIMITS.phi_max + 1e-6


def test_slew_limit_is_respected_when_enabled() -> None:
    """Torque cannot change faster than the configured physiological rate."""
    config = _config()
    result = DownswingOptimizer(config).solve("clubhead_speed")

    step = config.duration_s / (config.node_count - 1)
    max_change = np.max(np.abs(np.diff(result.torques, axis=0)), axis=0)
    allowed = np.asarray(config.torque_rate_limits) * step
    assert np.all(max_change <= allowed + 1e-6)
    assert result.max_slew_violation < 1e-6


def test_slew_limit_is_binding_and_changes_the_answer() -> None:
    """Removing the slew limit must produce a measurably faster, illegal swing.

    If it did not, the constraint would be decorative. The unconstrained
    optimum reverses hub torque between adjacent nodes to stop the arms dead at
    impact — mathematically optimal, physiologically impossible.
    """
    limited = DownswingOptimizer(_config()).solve("clubhead_speed")
    unlimited = DownswingOptimizer(_config(limit_torque_rate=False)).solve("clubhead_speed")

    assert unlimited.objective_value > limited.objective_value
    assert unlimited.max_slew_violation > limited.max_slew_violation


def test_loose_tolerance_regression_pin() -> None:
    """A loose ftol must not silently return the initial guess.

    SLSQP declares success as soon as the cost stops moving, which on an
    unscaled problem happens immediately after it finds feasibility. The
    optimizer therefore uses a tight tolerance by default; this test proves the
    default actually moves away from the starting trajectory.
    """
    optimizer = DownswingOptimizer(_config())
    guess_states, guess_torques = optimizer.initial_guess()
    result = optimizer.solve("clubhead_speed")

    assert not np.allclose(result.torques, guess_torques, atol=1e-3)
    assert not np.allclose(result.states, guess_states, atol=1e-3)


def test_scaling_is_what_makes_the_defects_small() -> None:
    """The non-dimensional decision vector is load-bearing, not cosmetic."""
    scaled = DownswingOptimizer(_config()).solve("clubhead_speed")
    unscaled = DownswingOptimizer(_config(use_variable_scaling=False)).solve("clubhead_speed")

    assert scaled.max_defect < _FEASIBLE_DEFECT
    assert scaled.max_defect < unscaled.max_defect


def test_result_exposes_signals_for_downstream_scoring() -> None:
    """The result carries the signals so every objective can rescore the swing."""
    result = DownswingOptimizer(_config()).solve("coriolis")

    assert result.signals.sample_count == _config().node_count
    assert np.allclose(result.signals.states, result.states)
    assert np.allclose(result.signals.torques, result.torques)


def test_infeasible_result_cannot_masquerade_as_an_optimum() -> None:
    """``feasible`` is derived from the defect, never from the solver's own flag."""
    result = DownswingOptimizer(_config()).solve("clubhead_speed")
    assert result.feasible == (result.max_defect < result.feasibility_tolerance)


def test_unknown_objective_is_rejected() -> None:
    """Fails closed rather than optimizing something unintended."""
    with pytest.raises(KeyError, match="Unknown swing objective"):
        DownswingOptimizer(_config()).solve("max_style_points")
