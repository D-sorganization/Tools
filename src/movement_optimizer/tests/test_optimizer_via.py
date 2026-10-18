# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Ordered, timed via points (:class:`ViaSchedule`) in the trajectory optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.models import make_deadlift_config
from movement_optimizer.trajectory import TrajectoryOptimizer, ViaPoint, ViaSchedule
from movement_optimizer.trajectory.optimizer_constraints import bar_knee_clearance
from movement_optimizer.trajectory.optimizer_via import (
    assemble_controls,
    clearance_times,
    via_initial_guess,
    via_layout,
)

Q0, Q1 = np.zeros(3), np.ones(3)


def _schedule(*fracs: float, end: float | None = None) -> ViaSchedule:
    return ViaSchedule(
        tuple(ViaPoint(f, np.full(3, 10.0 * (i + 1)), f"v{i}") for i, f in enumerate(fracs)),
        clearance_end_frac=end,
    )


def test_via_point_and_schedule_contracts() -> None:
    with pytest.raises(ValueError, match="t_frac"):
        ViaPoint(0.0, Q0)
    with pytest.raises(ValueError, match="finite"):
        ViaPoint(0.5, np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(ValueError, match="increasing"):
        _schedule(0.5, 0.3)
    with pytest.raises(ValueError, match="at least one"):
        ViaSchedule(())
    with pytest.raises(ValueError, match="clearance_end_frac"):
        _schedule(0.5, end=1.5)


def test_layout_pins_vias_at_their_times_and_spreads_free_points() -> None:
    schedule = _schedule(0.3, 0.45, 0.55)
    layout = via_layout(schedule, n_waypoints=12, duration=2.0)
    assert layout.t_ctrl.size == 12 + 2 + 3
    assert np.all(np.diff(layout.t_ctrl) > 0.0)
    np.testing.assert_allclose(layout.t_ctrl[layout.via_index], [0.6, 0.9, 1.1])
    assert layout.free_index.size == 12
    assert layout.t_ctrl[0] == 0.0 and layout.t_ctrl[-1] == 2.0


def test_controls_and_initial_guess_follow_the_schedule() -> None:
    schedule = _schedule(0.25, 0.75)
    layout = via_layout(schedule, n_waypoints=6, duration=1.0)
    wp = via_initial_guess(Q0, Q1, schedule, layout)
    assert wp.shape == (6, 3)
    q_all = assemble_controls(wp, Q0, Q1, schedule, layout)
    np.testing.assert_allclose(q_all[layout.via_index], schedule.configurations)
    knots = layout.t_ctrl[np.r_[0, layout.via_index, -1]]
    path = np.vstack([Q0, schedule.configurations, Q1])
    np.testing.assert_allclose(
        wp[:, 0], np.interp(layout.t_ctrl[layout.free_index], knots, path[:, 0])
    )


def test_clearance_window_ends_at_the_schedule_fraction() -> None:
    t_eval = np.linspace(0.0, 2.0, 11)
    assert clearance_times(t_eval, None, 2.0) is t_eval
    assert clearance_times(t_eval, _schedule(0.5), 2.0) is t_eval
    np.testing.assert_allclose(clearance_times(t_eval, _schedule(0.5, end=0.5), 2.0), t_eval[:6])


def test_optimizer_spline_passes_through_every_via(default_body) -> None:
    dyn, q_start, q_end, q_bounds = make_deadlift_config(default_body, 60.0)
    mid = 0.5 * (q_start + q_end)
    schedule = ViaSchedule((ViaPoint(0.3, mid, "a"), ViaPoint(0.6, q_end, "b")), 0.6)
    opt = TrajectoryOptimizer(
        default_body, dyn, "deadlift", 60.0, q_start, q_end, q_bounds,
        q_via=schedule, duration=2.0, n_waypoints=8, n_eval=41,
    )  # fmt: skip
    spline = opt.build_splines(opt._initial_guess().ravel())
    np.testing.assert_allclose(spline(0.6), mid, atol=1e-12)
    np.testing.assert_allclose(spline(1.2), q_end, atol=1e-12)
    assert opt._perturbed_guess(3).shape == (8, 3)
    clearance = [c for c in opt._build_constraints() if c["fun"] is bar_knee_clearance]
    assert clearance[0]["args"][1][-1] <= 1.2 + 1e-12


def test_legacy_single_via_grid_is_unchanged(default_body) -> None:
    dyn, q_start, q_end, q_bounds = make_deadlift_config(default_body, 60.0)
    via = 0.5 * (q_start + q_end)
    opt = TrajectoryOptimizer(
        default_body, dyn, "deadlift", 60.0, q_start, q_end, q_bounds,
        q_via=via, duration=2.0, n_waypoints=8, n_eval=41,
    )  # fmt: skip
    np.testing.assert_allclose(opt.t_ctrl, np.linspace(0.0, 2.0, 11))
    clearance = [c for c in opt._build_constraints() if c["fun"] is bar_knee_clearance]
    assert clearance[0]["args"][1] is opt.t_eval
