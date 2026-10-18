# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Clean and jerk full-lift configuration, timed via schedule and phase-correct bar.

UpstreamDrift #11745 (LIFT-5) packages its clean-and-jerk bundle from
``make_clean_config``, which models only the clean (floor setup -> standing
front rack through one mid-time via point) with a bar model that holds the bar
on the shoulders for the whole lift.

These tests solve the full clean and jerk with the UpstreamDrift biomechanical
parity standard (``exercises.clean_and_jerk``) and verify that:
- the bar hangs in the hands at the floor setup (~0.201 m, not 0.890 m);
- the shoulder and bar rise through the pulls;
- the lifter drops into a front-squat catch holding the standard's rack catch
  (knee 97.2 deg, trunk 18.3 deg);
- the lifter recovers to front rack, dips, drives, and catches the bar overhead;
- the bar ends locked out overhead (>2.0 m).
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.exercises import make_clean_and_jerk_config
from movement_optimizer.exercises.clean_and_jerk import (
    CLEAN_AND_JERK_LOCKOUT_KNEE_FLEX_DEG,
    CLEAN_AND_JERK_PHASES,
    CLEAN_AND_JERK_RACK_CATCH_KNEE_FLEX_DEG,
    CLEAN_AND_JERK_RACK_CATCH_TRUNK_DEG,
    clean_and_jerk_bar_path,
    clean_and_jerk_bar_weights,
)
from movement_optimizer.models import BodyModel
from movement_optimizer.trajectory import TrajectoryOptimizer, ViaSchedule

STANDARD_CATCH_PELVIS_PITCH_RAD = 0.3195
STANDARD_CATCH_LUMBAR_FLEX_RAD = -0.0486
CATCH_TRUNK_TOL_DEG = float(np.degrees(abs(STANDARD_CATCH_LUMBAR_FLEX_RAD)))


@pytest.fixture(scope="module")
def solved() -> dict[str, np.ndarray]:
    """The UpstreamDrift clean-and-jerk solve (80 kg, 1.78 m, 100 kg bar, 2.0 s, 151 samples)."""
    body = BodyModel(body_mass=80.0, height=1.78)
    dyn, q_start, q_end, q_bounds, q_via = make_clean_and_jerk_config(body, 100.0)
    result = TrajectoryOptimizer(
        body,
        dyn,
        "clean_and_jerk",
        100.0,
        q_start,
        q_end,
        q_bounds,
        q_via=q_via,
        duration=2.0,
        n_waypoints=14,
        n_eval=151,
    ).optimize()
    assert result.success
    q = np.asarray(result.q, dtype=float)
    t = np.asarray(result.t, dtype=float)
    frac = t / t[-1]
    shoulder = np.array([dyn.forward_kinematics(row)["shoulder"][1] for row in q])
    return {
        "q": q,
        "frac": frac,
        "shoulder": shoulder,
        "bar": clean_and_jerk_bar_path(dyn, q, frac),
    }


def _window(frac: np.ndarray, start: str, end: str) -> np.ndarray:
    return (frac >= CLEAN_AND_JERK_PHASES[start] - 1e-12) & (
        frac <= CLEAN_AND_JERK_PHASES[end] + 1e-12
    )


def test_shoulder_rises_through_the_pull(solved) -> None:
    pull = _window(solved["frac"], "first_pull", "second_pull")
    shoulder = solved["shoulder"][pull]
    assert np.all(np.diff(shoulder) >= -1e-4), "the shoulder must rise through the pulls"
    assert shoulder[-1] > solved["shoulder"][0] + 0.2


def test_bar_rises_through_the_pull(solved) -> None:
    pull = _window(solved["frac"], "first_pull", "second_pull")
    assert np.all(np.diff(solved["bar"][pull, 1]) >= -1e-4)


def test_lifter_drops_under_the_bar_into_rack_catch(solved) -> None:
    second_pull = int(np.argmin(np.abs(solved["frac"] - CLEAN_AND_JERK_PHASES["second_pull"])))
    rack_catch = int(np.argmin(np.abs(solved["frac"] - CLEAN_AND_JERK_PHASES["rack_catch"])))
    assert solved["shoulder"][rack_catch] < solved["shoulder"][second_pull] - 0.05


def test_bar_is_not_on_shoulders_at_setup(solved) -> None:
    # At setup, bar hangs in the hands (~0.201 m), well below the shoulder (~0.890 m)
    assert solved["bar"][0, 1] < solved["shoulder"][0] - 0.5
    w_hang, w_rack, w_oh = clean_and_jerk_bar_weights(np.array([0.0]))
    assert w_hang[0] == 1.0
    assert w_rack[0] == 0.0
    assert w_oh[0] == 0.0


def test_bar_hangs_in_pull_rests_in_rack_and_is_overhead_from_jerk_catch(solved) -> None:
    frac, bar, shoulder = solved["frac"], solved["bar"], solved["shoulder"]
    pull = frac < CLEAN_AND_JERK_PHASES["second_pull"]
    rack = (frac >= CLEAN_AND_JERK_PHASES["rack_catch"]) & (
        frac <= CLEAN_AND_JERK_PHASES["rack_recovery"]
    )
    overhead = frac >= CLEAN_AND_JERK_PHASES["overhead_catch"]

    # In the pull, bar hangs below shoulder
    assert np.all(bar[pull, 1] < shoulder[pull])
    # In the rack, bar sits at shoulder height
    assert np.all(np.abs(bar[rack, 1] - shoulder[rack]) < 0.05)
    # Overhead, bar is locked out above shoulder
    assert np.all(bar[overhead, 1] > shoulder[overhead] + 0.5)


def test_bar_goes_overhead_to_lockout(solved) -> None:
    # Bar must reach overhead lockout height (>2.0 m for 1.78 m lifter)
    assert np.max(solved["bar"][:, 1]) > 2.0
    assert solved["bar"][-1, 1] > 2.0


def test_rack_catch_trunk_lean_matches_the_standard(solved) -> None:
    standard = np.degrees(STANDARD_CATCH_PELVIS_PITCH_RAD)
    trunk = np.degrees(
        np.interp(CLEAN_AND_JERK_PHASES["rack_catch"], solved["frac"], solved["q"][:, 2])
    )
    assert abs(trunk - standard) <= CATCH_TRUNK_TOL_DEG
    assert pytest.approx(standard) == CLEAN_AND_JERK_RACK_CATCH_TRUNK_DEG


def test_config_rack_catch_holds_the_standard_trunk_and_knee() -> None:
    body = BodyModel(body_mass=80.0, height=1.78)
    _dyn, _qs, _qe, q_bounds, q_via = make_clean_and_jerk_config(body, 100.0)
    assert isinstance(q_via, ViaSchedule)
    catch = {p.name: p.q for p in q_via.points}["rack_catch"]
    assert np.degrees(catch[2]) == pytest.approx(CLEAN_AND_JERK_RACK_CATCH_TRUNK_DEG)
    assert np.degrees(catch[1] - catch[0]) == pytest.approx(CLEAN_AND_JERK_RACK_CATCH_KNEE_FLEX_DEG)
    assert np.all((catch >= q_bounds[:, 0]) & (catch <= q_bounds[:, 1]))


def test_config_lockout_holds_the_standard_knee() -> None:
    body = BodyModel(body_mass=80.0, height=1.78)
    _dyn, _qs, qe, q_bounds, _q_via = make_clean_and_jerk_config(body, 100.0)
    assert np.degrees(qe[1] - qe[0]) == pytest.approx(CLEAN_AND_JERK_LOCKOUT_KNEE_FLEX_DEG)
    assert np.all((qe >= q_bounds[:, 0]) & (qe <= q_bounds[:, 1]))


def test_via_points_sit_at_the_standard_phase_times() -> None:
    body = BodyModel(body_mass=80.0, height=1.78)
    *_, q_via = make_clean_and_jerk_config(body, 100.0)
    for point in q_via.points:
        assert point.t_frac == CLEAN_AND_JERK_PHASES[point.name]
    assert q_via.clearance_end_frac == CLEAN_AND_JERK_PHASES["second_pull"]


def test_bar_weights_partition_of_unity() -> None:
    t = np.linspace(0.0, 1.0, 101)
    w_h, w_r, w_o = clean_and_jerk_bar_weights(t)
    np.testing.assert_allclose(w_h + w_r + w_o, 1.0, atol=1e-12)
    assert np.all(w_h >= 0.0) and np.all(w_r >= 0.0) and np.all(w_o >= 0.0)


def test_bar_path_rejects_mismatched_times() -> None:
    body = BodyModel(body_mass=80.0, height=1.78)
    dyn, q_start, *_ = make_clean_and_jerk_config(body, 100.0)
    with pytest.raises(ValueError, match="one entry per"):
        clean_and_jerk_bar_path(dyn, np.vstack([q_start, q_start]), np.array([0.0]))
