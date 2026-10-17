# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""The snatch has a real pull, a phase-correct bar and the standard's catch.

The previous configuration interpolated from the floor setup straight to
the overhead-squat catch through one mid-time via point: with the 2.0 s
solve UpstreamDrift uses (80 kg, 1.78 m, 100 kg bar, 151 samples) the
shoulder fell from 0.876 m to 0.653 m in the first 0.93 s instead of rising,
the bar model held the bar overhead at the floor (1.565 m at the setup) and
the catch trunk leaned 59.3 deg against the standard's 18.7 deg.

These tests solve that same problem and check the phases of the
UpstreamDrift biomechanical parity standard (``exercises.snatch``).
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.exercises import make_snatch_config
from movement_optimizer.exercises.snatch import (
    SNATCH_CATCH_KNEE_FLEX_DEG,
    SNATCH_CATCH_TRUNK_DEG,
    SNATCH_PHASES,
    snatch_bar_path,
    snatch_bar_weight,
)
from movement_optimizer.models import BodyModel
from movement_optimizer.trajectory import TrajectoryOptimizer, ViaSchedule

#: The standard's catch: pelvis pitch 0.3264 rad, lumbar flexion -0.0675 rad.
STANDARD_CATCH_PELVIS_PITCH_RAD = 0.3264
STANDARD_CATCH_LUMBAR_FLEX_RAD = -0.0675
#: The planar model has one trunk segment; the standard's pelvis and thorax
#: differ by its catch lumbar flexion (3.87 deg), so that is the tolerance.
CATCH_TRUNK_TOL_DEG = float(np.degrees(abs(STANDARD_CATCH_LUMBAR_FLEX_RAD)))


@pytest.fixture(scope="module")
def solved() -> dict[str, np.ndarray]:
    """The UpstreamDrift LIFT-5 snatch solve (2.0 s, 151 samples)."""
    body = BodyModel(body_mass=80.0, height=1.78)
    dyn, q_start, q_end, q_bounds, q_via = make_snatch_config(body, 100.0)
    result = TrajectoryOptimizer(
        body,
        dyn,
        "snatch",
        100.0,
        q_start,
        q_end,
        q_bounds,
        q_via=q_via,
        duration=2.0,
        n_waypoints=12,
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
        "bar": snatch_bar_path(dyn, q, frac),
    }


def _window(frac: np.ndarray, start: str, end: str) -> np.ndarray:
    return (frac >= SNATCH_PHASES[start] - 1e-12) & (frac <= SNATCH_PHASES[end] + 1e-12)


def test_shoulder_rises_monotonically_through_the_pull(solved) -> None:
    pull = _window(solved["frac"], "first_pull", "turnover")
    shoulder = solved["shoulder"][pull]
    assert np.all(np.diff(shoulder) >= 0.0), "the shoulder must rise through both pulls"
    # The pull ends at the highest shoulder of the lift before the drop.
    assert shoulder[-1] > solved["shoulder"][0] + 0.3


def test_bar_rises_monotonically_through_the_pull(solved) -> None:
    pull = _window(solved["frac"], "first_pull", "turnover")
    assert np.all(np.diff(solved["bar"][pull, 1]) >= 0.0)


def test_lifter_drops_under_the_bar_after_the_pull(solved) -> None:
    turnover = int(np.argmin(np.abs(solved["frac"] - SNATCH_PHASES["turnover"])))
    catch = int(np.argmin(np.abs(solved["frac"] - SNATCH_PHASES["catch"])))
    assert solved["shoulder"][catch] < solved["shoulder"][turnover] - 0.1


def test_bar_is_not_overhead_at_the_setup(solved) -> None:
    assert solved["bar"][0, 1] < solved["shoulder"][0]
    assert snatch_bar_weight(np.array([0.0]))[0] == 0.0


def test_bar_hangs_through_the_pull_and_is_overhead_from_the_catch(solved) -> None:
    frac, bar, shoulder = solved["frac"], solved["bar"], solved["shoulder"]
    pull = frac < SNATCH_PHASES["turnover"]
    overhead = frac >= SNATCH_PHASES["catch"]
    assert np.all(bar[pull, 1] < shoulder[pull])
    assert np.all(bar[overhead, 1] > shoulder[overhead])


def test_catch_trunk_lean_matches_the_standard(solved) -> None:
    standard = np.degrees(STANDARD_CATCH_PELVIS_PITCH_RAD)
    trunk = np.degrees(np.interp(SNATCH_PHASES["catch"], solved["frac"], solved["q"][:, 2]))
    assert abs(trunk - standard) <= CATCH_TRUNK_TOL_DEG
    assert pytest.approx(standard) == SNATCH_CATCH_TRUNK_DEG


def test_config_catch_holds_the_standard_trunk_and_knee() -> None:
    body = BodyModel(body_mass=80.0, height=1.78)
    _dyn, _qs, _qe, q_bounds, q_via = make_snatch_config(body, 100.0)
    assert isinstance(q_via, ViaSchedule)
    catch = {p.name: p.q for p in q_via.points}["catch"]
    assert np.degrees(catch[2]) == pytest.approx(SNATCH_CATCH_TRUNK_DEG)
    assert np.degrees(catch[1] - catch[0]) == pytest.approx(SNATCH_CATCH_KNEE_FLEX_DEG)
    assert np.all((catch >= q_bounds[:, 0]) & (catch <= q_bounds[:, 1]))


def test_via_points_sit_at_the_standard_phase_times() -> None:
    body = BodyModel(body_mass=80.0, height=1.78)
    *_, q_via = make_snatch_config(body, 100.0)
    for point in q_via.points:
        assert point.t_frac == SNATCH_PHASES[point.name]
    assert q_via.clearance_end_frac == SNATCH_PHASES["turnover"]


def test_bar_path_rejects_mismatched_times() -> None:
    body = BodyModel(body_mass=80.0, height=1.78)
    dyn, q_start, *_ = make_snatch_config(body, 100.0)
    with pytest.raises(ValueError, match="one entry per"):
        snatch_bar_path(dyn, np.vstack([q_start, q_start]), np.array([0.0]))
