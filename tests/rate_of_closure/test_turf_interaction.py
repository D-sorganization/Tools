"""Rate-to-shared-turf adapter contracts."""

from __future__ import annotations

import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    SimulationConfig,
    run_simulation,
    turf_interaction_for_run,
)
from rate_of_closure.simulation.records import SimulationRun
from shared.python.golf_club import (
    GroundPlane,
    TurfContactStatus,
    TurfPreset,
    WedgePreset,
    turf_profile_preset,
    wedge_preset,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _run() -> SimulationRun:
    return run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=get_club("Sand Wedge"),
            impact_time_s=0.03,
        )
    )


def test_adapter_preserves_no_ground_contact_as_typed_unavailability() -> None:
    snapshot = turf_interaction_for_run(
        _run(),
        wedge_preset(WedgePreset.MID_BOUNCE),
        GroundPlane(point_m=(0.0, -10.0, 0.0)),
        turf_profile_preset(TurfPreset.FIRM_FAIRWAY),
    )

    assert snapshot.first_contact_wrench is None
    assert snapshot.reduced_contact is None
    assert "does not replay" in snapshot.limitations


def test_adapter_consumes_registered_pose_twist_and_selected_head_mass() -> None:
    run = _run()
    snapshot = turf_interaction_for_run(
        run,
        wedge_preset(WedgePreset.MID_BOUNCE),
        GroundPlane(point_m=(0.0, -0.015, 0.0)),
        turf_profile_preset(TurfPreset.SOFT_TURF),
    )

    assert snapshot.first_contact_wrench is not None
    assert snapshot.first_contact_wrench.active_patches
    assert snapshot.reduced_contact is not None
    assert snapshot.reduced_contact.status in {
        TurfContactStatus.NO_CONTACT,
        TurfContactStatus.SEPARATED,
    }
    event = snapshot.ground_clearance.analysis.first_ground_contact
    assert event is not None
    assert snapshot.reduced_contact.initial_kinetic_energy_j == pytest.approx(
        0.5
        * run.config.club.head_mass_kg
        * sum(
            component**2
            for component in (
                *event.tangential_velocity_mps,
                event.normal_velocity_mps,
            )
        )
    )
