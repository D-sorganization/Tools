"""Contract tests for the objective realism ranking.

The module-level fixtures deliberately use the **mis-specified** club (#4785) so
the artifact stays characterised. The corrected-club test at the bottom is the
one that answers the actual question, and it is where the useful result lives:
with an inertia-matched club, four objectives tie and centrifugal release
impulse is measurably worse.

Closes #4780, #4785.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import JointLimits, PendulumParams, TorqueClamp
from double_pendulum_golf.swing_objectives.downswing import DownswingConfig
from double_pendulum_golf.swing_objectives.objective_realism import (
    ObjectiveRealism,
    ObjectiveRealismRanking,
    objective_realism_ranking,
)
from double_pendulum_golf.swing_objectives.reference_kinematics import (
    TOUR_DRIVER_BANDS,
)

_PARAMS = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10, mClub=0.20)
_SUBSET = ("clubhead_speed", "centrifugal", "coriolis")


def _config(min_hand_speed_ms: float | None = None) -> DownswingConfig:
    """Baseline with the **mis-specified** club (#4785), optionally floored."""
    return DownswingConfig(
        params=_PARAMS,
        node_count=17,
        duration_s=0.36,
        initial_state=np.array([2.618, 1.745, 0.0, 0.0]),
        impact_theta1_rad=0.0,
        torque_clamp=TorqueClamp(max_torque1=250.0, max_torque2=20.0),
        joint_limits=JointLimits(
            phi_min=-0.175, phi_max=2.094, theta1_min=-4.0, theta1_max=4.0
        ),
        min_hand_speed_ms=min_hand_speed_ms,
    )


@pytest.fixture(scope="module")
def ranking() -> ObjectiveRealismRanking:
    """One ranking reused across the module; each entry is a full solve."""
    return objective_realism_ranking(_config(), objective_keys=_SUBSET)


def test_ranking_is_sorted_most_golf_like_first(ranking) -> None:
    """Callers read entries positionally, so the order is part of the contract."""
    deviations = [entry.total_deviation for entry in ranking.entries]
    assert deviations == sorted(deviations)
    assert ranking.best is ranking.entries[0]
    assert all(isinstance(entry, ObjectiveRealism) for entry in ranking.entries)


def test_no_objective_reaches_measured_behaviour_with_a_bad_club(ranking) -> None:
    """With 2.1x the real coupling, no objective looks like a real swing."""
    assert not ranking.reaches_measured_behaviour
    assert ranking.mean_deviation > 1.0
    for entry in ranking.entries:
        assert entry.inside_count < ranking.band_count


def test_unconstrained_every_objective_is_hopelessly_far_from_a_real_swing(
    ranking,
) -> None:
    """Without a hand-speed floor the ordering is between degrees of wrong.

    The objectives do separate here — Coriolis is markedly worse than the rest —
    but the best of them still sits tens of half-widths outside the measured
    bands, so the separation is between bad answers, not a finding about golf.
    """
    assert ranking.mean_deviation > 50.0
    assert ranking.best.total_deviation > 50.0


def test_the_ranking_reports_itself_as_non_discriminating_once_realistic() -> None:
    """The load-bearing negative result.

    In the only regime where the swings are anywhere near golf-like — hands held
    above 3 m/s — the objectives land within a fraction of a percent of each
    other. Reporting that ordering as "this is what golfers optimize" would be
    reading noise on top of a much larger model bias, so the ranking says so
    itself rather than leaving the caller to notice.
    """
    realistic = objective_realism_ranking(
        _config(min_hand_speed_ms=3.0), objective_keys=_SUBSET
    )
    assert not realistic.is_discriminating
    assert realistic.deviation_spread < 0.05 * realistic.mean_deviation


def test_the_club_arm_rate_ratio_is_what_fails(ranking) -> None:
    """Naming the dominant failure keeps the diagnosis attached to the number."""
    assert all(entry.worst_observable == "club_arm_rate_ratio" for entry in ranking.entries)


def test_a_hand_speed_floor_shrinks_the_gap_without_closing_it() -> None:
    """Constraining the hands helps a lot and is still not enough."""
    unconstrained = objective_realism_ranking(_config(), objective_keys=_SUBSET)
    constrained = objective_realism_ranking(
        _config(min_hand_speed_ms=3.0), objective_keys=_SUBSET
    )
    assert constrained.mean_deviation < 0.5 * unconstrained.mean_deviation
    assert not constrained.reaches_measured_behaviour
    assert not constrained.is_discriminating


def test_every_entry_reports_its_own_feasibility(ranking) -> None:
    """A ranking built from infeasible swings would be meaningless."""
    assert all(entry.feasible for entry in ranking.entries)


def test_observables_cover_every_reference_band(ranking) -> None:
    """Each entry must be scoreable on the full band set, not a convenient subset."""
    band_keys = {band.key for band in TOUR_DRIVER_BANDS}
    for entry in ranking.entries:
        assert band_keys <= set(entry.observables)


def test_rejects_an_empty_objective_list() -> None:
    """Contract: there must be something to rank."""
    with pytest.raises(ValueError, match="at least one objective"):
        objective_realism_ranking(_config(), objective_keys=[])


def test_rejects_an_unknown_objective() -> None:
    """Fails closed rather than silently ranking a smaller set."""
    with pytest.raises(KeyError, match="Unknown swing objective"):
        objective_realism_ranking(_config(), objective_keys=("clubhead_speed", "vibes"))


def test_an_inertia_matched_club_makes_the_ranking_meaningful() -> None:
    """The corrected result (#4785), and the answer to the epic's question.

    With the shipped preset's inertia-matched club every objective lands 5 of 6
    observables inside the measured bands, and the ranking separates: clubhead
    speed, Coriolis transfer, energy transfer and impulse transfer are
    indistinguishable, while **centrifugal release impulse is measurably worse**
    — it gives up around 1 m/s of clubhead speed and pushes the club/arm rate
    ratio to the edge of its band.
    """
    from double_pendulum_golf.swing_objectives.presets import build_config

    ranking = objective_realism_ranking(build_config())

    assert ranking.mean_deviation < 2.0
    assert all(entry.inside_count >= 4 for entry in ranking.entries)
    assert all(entry.feasible for entry in ranking.entries)

    scores = {entry.key: entry.total_deviation for entry in ranking.entries}
    assert scores["centrifugal"] > scores["clubhead_speed"]
    assert scores["coriolis"] == pytest.approx(scores["clubhead_speed"], abs=0.05)

    speeds = {entry.key: entry.clubhead_speed_ms for entry in ranking.entries}
    assert speeds["clubhead_speed"] - speeds["centrifugal"] > 0.5
