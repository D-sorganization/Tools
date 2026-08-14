"""Frame-safe selected assembly binding to impact-solver adapter tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rate_of_closure.club import (
    ClubAssemblyBinding,
    get_club,
    parse_club_assembly_binding,
)
from rate_of_closure.club.simulation_adapter import (
    APP_FRAME_ID,
    WorldFromHeadAttitude,
    adapt_club_assembly_for_impact,
)
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    SimulationConfig,
    run_simulation,
    run_to_json_dict,
)
from rate_of_closure.simulation.contact import ContactMode
from shared.python.swing_sim.ball_setup import BallSetup, BallSupportMode

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_DRIVER = "Driver 10.5\N{DEGREE SIGN}"
_FIXTURE = Path(__file__).parent / "fixtures" / "club_assembly_binding_driver_10_5.json"
_SCENARIO = ImpactScenario(clubhead_speed_mph=113.0)


def _binding() -> ClubAssemblyBinding:
    spec = get_club(_DRIVER)
    return parse_club_assembly_binding(spec, _FIXTURE.read_bytes())


def test_unbound_adapter_preserves_scalar_path_and_reports_every_gap() -> None:
    adapted = adapt_club_assembly_for_impact(get_club(_DRIVER), None, None)

    assert adapted.head_mass_kg == pytest.approx(0.2)
    assert adapted.head_inertia_tensor_app_kg_m2 is None
    assert adapted.head_inertia.status == "unavailable"
    assert "binding" in adapted.head_inertia.reason
    assert adapted.head_center_of_mass.status == "unavailable"
    assert adapted.assembly_mass_properties.status == "unavailable"

    run = run_simulation(SimulationConfig(scenario=_SCENARIO, club=get_club(_DRIVER)))
    assert run.club_assembly_usage.head_inertia.status == "unavailable"


def test_adapter_rotates_only_head_tensor_with_explicit_frame_attitude() -> None:
    binding = _binding()
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    attitude = WorldFromHeadAttitude(
        from_frame_id="rate_of_closure.head",
        to_frame_id=APP_FRAME_ID,
        rotation=rotation,
        provenance="contract-test-attitude",
    )
    assert rotation.flags.writeable
    assert not attitude.rotation.flags.writeable

    adapted = adapt_club_assembly_for_impact(get_club(_DRIVER), binding, attitude)
    head = binding.head_properties_in_selected_frame()
    expected = rotation @ np.asarray(head.inertia_at_com_kg_m2) @ rotation.T

    assert adapted.head_inertia.status == "available"
    np.testing.assert_allclose(adapted.head_inertia_tensor_app_kg_m2, expected)
    assert adapted.head_center_of_mass.status == "unavailable"
    assert "does not accept a full head-CG vector" in adapted.head_center_of_mass.reason
    assert adapted.assembly_mass_properties.status == "unavailable"
    assert "must not substitute" in adapted.assembly_mass_properties.reason
    assert not np.allclose(
        adapted.head_inertia_tensor_app_kg_m2,
        np.asarray(binding.assembly.mass_properties.inertia_at_com_kg_m2),
    )


def test_adapter_rejects_selection_and_frame_mismatch() -> None:
    binding = _binding()
    with pytest.raises(ValueError, match="selected ClubSpec identity"):
        adapt_club_assembly_for_impact(
            replace(get_club(_DRIVER), loft_deg=11.0), binding, None
        )

    with pytest.raises(ValueError, match="from rate_of_closure.head"):
        WorldFromHeadAttitude(
            from_frame_id="unknown.head",
            to_frame_id=APP_FRAME_ID,
            rotation=np.eye(3),
            provenance="bad-frame",
        )


def test_manual_pipeline_uses_bound_head_tensor_but_miss_skips_impact() -> None:
    binding = _binding()
    hit = run_simulation(
        SimulationConfig(
            scenario=_SCENARIO, club=get_club(_DRIVER), assembly_binding=binding
        )
    )

    assert hit.club_assembly_usage.head_inertia.status == "available"
    assert hit.club_assembly_usage.head_inertia.consumed is True
    exported_usage = run_to_json_dict(hit)["club_assembly_usage"]
    assert exported_usage["head_inertia"]["status"] == "available"
    assert "head_inertia_tensor_app_kg_m2" not in exported_usage

    miss = run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=get_club(_DRIVER),
            assembly_binding=binding,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
            ball_setup=BallSetup(BallSupportMode.TEE, 0.10),
        )
    )
    assert miss.impact_outcome.status.value == "miss"
    assert miss.club_assembly_usage.head_inertia.status == "not_used"
    assert "no club-ball impact" in miss.club_assembly_usage.head_inertia.reason


def test_pendulum_binding_does_not_infer_selected_head_attitude() -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=get_club(_DRIVER),
            assembly_binding=_binding(),
            source_kind="double_pendulum",
            swing_duration_s=0.05,
        )
    )

    assert run.club_assembly_usage.head_inertia.status == "unavailable"
    assert "does not declare" in run.club_assembly_usage.head_inertia.reason
