# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""The clean and jerk is one credible lift: pull, rack catch, jerk, lockout.

The ``clean`` configuration runs from the floor setup to a standing front
rack through one mid-time via point, with the ``clean`` bar model holding the
bar on the shoulders for the whole lift.  With the solve UpstreamDrift used
for its clean-and-jerk bundle (80 kg, 1.78 m, 100 kg bar, 1.2 s, 151
samples) the bar sits at 0.890 m at the floor setup (on the shoulders, not
in the hands at 0.201 m), the knee never flexes past its 90.5 deg setup
flexion, so there is no front-squat catch (the standard's 97.2 deg), and the
bar never goes overhead (its highest point is the 1.347 m standing shoulder).

These tests solve the full clean-and-jerk configuration and check the phases
of the UpstreamDrift biomechanical parity standard
(``exercises.clean_and_jerk``).
"""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.exercises import make_clean_and_jerk_config
from movement_optimizer.exercises.clean_and_jerk import (
    CLEAN_AND_JERK_EVENTS,
    CLEAN_AND_JERK_PHASES,
    OVERHEAD_CATCH_TRUNK_DEG,
    RACK_CATCH_KNEE_FLEX_DEG,
    RACK_CATCH_TRUNK_DEG,
    clean_and_jerk_bar_path,
    clean_and_jerk_bar_weights,
)
from movement_optimizer.models import BodyModel
from movement_optimizer.trajectory import TrajectoryOptimizer, ViaSchedule
from movement_optimizer.trajectory.optimizer_constraints import bar_knee_clearance

#: The standard's phase starts (``exercises.clean_and_jerk.phases``).
STANDARD_PHASES = {
    "setup": 0.0,
    "first_pull": 0.08,
    "second_pull": 0.2,
    "rack_catch": 0.3,
    "rack_recovery": 0.4,
    "jerk_dip_drive": 0.55,
    "overhead_catch": 0.7,
    "overhead_recovery": 0.85,
}
#: The standard's rack catch: pelvis pitch 0.3195 rad, lumbar flexion
#: -0.0486 rad.  The planar trunk is one segment, so the tolerance is the
#: standard's lumbar flexion (pelvis to thorax) at that pose.
STANDARD_RACK_CATCH_PITCH_RAD = 0.3195
RACK_CATCH_TRUNK_TOL_DEG = float(np.degrees(0.0486))
#: The standard has no overhead-catch pose; its lockout (pelvis pitch
#: -0.0009 rad, lumbar flexion 0.0635 rad) is the overhead trunk reference.
STANDARD_LOCKOUT_PITCH_RAD = -0.0009
OVERHEAD_TRUNK_TOL_DEG = float(np.degrees(0.0635))
DURATION_S = 4.0


def _at(frac: np.ndarray, phase: str) -> int:
    t = {**CLEAN_AND_JERK_PHASES, **CLEAN_AND_JERK_EVENTS}[phase]
    return int(np.argmin(np.abs(frac - t)))


def _window(frac: np.ndarray, start: float, end: float) -> np.ndarray:
    return (frac >= start - 1e-12) & (frac <= end + 1e-12)


@pytest.fixture(scope="module")
def body() -> BodyModel:
    return BodyModel(body_mass=80.0, height=1.78)


@pytest.fixture(scope="module")
def solved(body: BodyModel) -> dict[str, np.ndarray]:
    """The UpstreamDrift LIFT-5 clean-and-jerk solve (4.0 s, 151 samples)."""
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
        duration=DURATION_S,
        n_waypoints=12,
        n_eval=151,
    ).optimize()
    assert result.success
    q = np.asarray(result.q, dtype=float)
    frac = np.asarray(result.t, dtype=float) / DURATION_S
    shoulder = np.array([dyn.forward_kinematics(row)["shoulder"][1] for row in q])
    return {
        "q": q,
        "frac": frac,
        "shoulder": shoulder,
        "bar": clean_and_jerk_bar_path(dyn, q, frac),
    }


def test_phases_are_the_standards() -> None:
    assert dict(CLEAN_AND_JERK_PHASES) == STANDARD_PHASES
    events = CLEAN_AND_JERK_EVENTS
    assert STANDARD_PHASES["second_pull"] < events["turnover"] < STANDARD_PHASES["rack_catch"]
    dip_drive = (STANDARD_PHASES["jerk_dip_drive"], STANDARD_PHASES["overhead_catch"])
    assert dip_drive[0] < events["dip_bottom"] < events["drive_extension"] < dip_drive[1]


def test_bar_is_not_on_the_shoulders_at_the_setup(solved, body: BodyModel) -> None:
    # Hanging in the hands, one arm length below the shoulder.
    assert solved["bar"][0, 1] == pytest.approx(solved["shoulder"][0] - body.L_arm)
    assert clean_and_jerk_bar_weights(np.array([0.0]))[0][0] == 0.0


def test_shoulder_and_bar_rise_through_the_pull(solved) -> None:
    pull = _window(
        solved["frac"], CLEAN_AND_JERK_PHASES["first_pull"], CLEAN_AND_JERK_EVENTS["turnover"]
    )
    assert np.all(np.diff(solved["shoulder"][pull]) >= 0.0)
    assert np.all(np.diff(solved["bar"][pull, 1]) >= 0.0)
    assert solved["shoulder"][pull][-1] > solved["shoulder"][0] + 0.3


def test_lifter_drops_under_the_bar_into_a_front_squat(solved) -> None:
    turnover = _at(solved["frac"], "turnover")
    catch = _at(solved["frac"], "rack_catch")
    assert solved["shoulder"][catch] < solved["shoulder"][turnover] - 0.2
    knee = np.degrees(solved["q"][catch, 1] - solved["q"][catch, 0])
    assert knee == pytest.approx(RACK_CATCH_KNEE_FLEX_DEG, abs=1e-6)


def test_bar_is_at_front_rack_height_from_the_catch_to_the_drive(solved) -> None:
    frac, bar, shoulder = solved["frac"], solved["bar"], solved["shoulder"]
    rack = _window(frac, CLEAN_AND_JERK_PHASES["rack_catch"], CLEAN_AND_JERK_EVENTS["drive_extension"])
    np.testing.assert_allclose(bar[rack, 1], shoulder[rack], atol=1e-12)


def test_jerk_dips_then_drives(solved) -> None:
    standing = _at(solved["frac"], "jerk_dip_drive")
    dip = _at(solved["frac"], "dip_bottom")
    drive = _at(solved["frac"], "drive_extension")
    assert solved["shoulder"][dip] < solved["shoulder"][standing] - 0.05
    assert solved["shoulder"][drive] > solved["shoulder"][dip] + 0.05


def test_bar_is_overhead_from_the_jerk_catch_to_the_lockout(solved, body: BodyModel) -> None:
    frac, bar, shoulder = solved["frac"], solved["bar"], solved["shoulder"]
    overhead = frac >= CLEAN_AND_JERK_PHASES["overhead_catch"]
    np.testing.assert_allclose(bar[overhead, 1], shoulder[overhead] + body.L_arm, atol=1e-12)
    # Locked out standing: the shoulder is near its full standing height.
    assert shoulder[-1] > 0.95 * body.L.sum()


def test_catch_trunk_leans_match_the_standard(solved) -> None:
    frac, trunk = solved["frac"], np.degrees(solved["q"][:, 2])
    rack = float(np.interp(CLEAN_AND_JERK_PHASES["rack_catch"], frac, trunk))
    overhead = float(np.interp(CLEAN_AND_JERK_PHASES["overhead_catch"], frac, trunk))
    rack_std = float(np.degrees(STANDARD_RACK_CATCH_PITCH_RAD))
    overhead_std = float(np.degrees(STANDARD_LOCKOUT_PITCH_RAD))
    assert abs(rack - rack_std) <= RACK_CATCH_TRUNK_TOL_DEG
    assert abs(overhead - overhead_std) <= OVERHEAD_TRUNK_TOL_DEG
    assert pytest.approx(rack_std) == RACK_CATCH_TRUNK_DEG
    assert pytest.approx(overhead_std) == OVERHEAD_CATCH_TRUNK_DEG


def test_via_points_sit_at_the_standard_phase_times(body: BodyModel) -> None:
    *_, q_bounds, q_via = make_clean_and_jerk_config(body, 100.0)
    assert isinstance(q_via, ViaSchedule)
    times = {**CLEAN_AND_JERK_PHASES, **CLEAN_AND_JERK_EVENTS}
    for point in q_via.points:
        assert point.t_frac == times[point.name]
        assert np.all((point.q >= q_bounds[:, 0]) & (point.q <= q_bounds[:, 1])), point.name
    assert q_via.clearance_end_frac == CLEAN_AND_JERK_EVENTS["turnover"]


def test_clean_and_jerk_is_a_pull_with_bar_knee_clearance(body: BodyModel) -> None:
    dyn, q_start, q_end, q_bounds, q_via = make_clean_and_jerk_config(body, 100.0)
    opt = TrajectoryOptimizer(
        body, dyn, "clean_and_jerk", 100.0, q_start, q_end, q_bounds,
        q_via=q_via, duration=DURATION_S, n_waypoints=12, n_eval=151,
    )  # fmt: skip
    clearance = [c for c in opt._build_constraints() if c["fun"] is bar_knee_clearance]
    assert len(clearance) == 1
    assert clearance[0]["args"][1][-1] <= CLEAN_AND_JERK_EVENTS["turnover"] * DURATION_S + 1e-12


def test_bar_path_rejects_mismatched_times(body: BodyModel) -> None:
    dyn, q_start, *_ = make_clean_and_jerk_config(body, 100.0)
    with pytest.raises(ValueError, match="one entry per"):
        clean_and_jerk_bar_path(dyn, np.vstack([q_start, q_start]), np.array([0.0]))
