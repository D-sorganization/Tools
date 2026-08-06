"""Rate-to-shared-wedge ground-clearance adapter contracts."""

from __future__ import annotations

import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    ContactMode,
    SimulationConfig,
    ground_clearance_for_run,
    run_simulation,
)
from shared.python.golf_club import (
    ContactSequence,
    GroundPlane,
    WedgePreset,
    wedge_preset,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_adapter_uses_actual_ball_contact_for_a_hit() -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=get_club("Pitching Wedge"),
            impact_time_s=0.03,
        )
    )

    analysis = ground_clearance_for_run(
        run,
        wedge_preset(WedgePreset.MID_BOUNCE),
        GroundPlane(point_m=(0.0, -10.0, 0.0)),
    )

    assert analysis.ball_contact_time_s == pytest.approx(0.03)
    assert analysis.sequence is ContactSequence.BALL_ONLY
    assert analysis.first_ground_contact is None


def test_adapter_preserves_a_miss_without_fabricating_ball_contact() -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=get_club("Pitching Wedge"),
            source_kind="double_pendulum",
            swing_duration_s=0.05,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
        )
    )

    analysis = ground_clearance_for_run(
        run,
        wedge_preset(WedgePreset.LOW_BOUNCE),
        GroundPlane(point_m=(0.0, -10.0, 0.0)),
    )

    assert run.impact_time_s is None
    assert analysis.ball_contact_time_s is None
    assert analysis.sequence is ContactSequence.NO_CONTACT_MISS


def test_adapter_rejects_non_run_inputs() -> None:
    with pytest.raises(TypeError, match="SimulationRun"):
        ground_clearance_for_run(  # type: ignore[arg-type]
            object(),
            wedge_preset(WedgePreset.MID_BOUNCE),
            GroundPlane(),
        )
