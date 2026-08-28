"""Manual delivery and declared shaft-datum contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rate_of_closure.club import face_center_point, get_club, hosel_point
from rate_of_closure.model import ImpactScenario, impact_frame, solve
from rate_of_closure.simulation import (
    ManualDeliveryConfig,
    ManualSwingSource,
    ShaftAxisDatum,
    SimulationConfig,
    impact_kinematics_for_run,
    manual_delivery_from_json_dict,
    run_simulation,
    run_to_json_dict,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_WEDGE = get_club("Pitching Wedge")


def _forward_lean_rotation(lean_deg: float) -> np.ndarray:
    angle = math.radians(-lean_deg)
    return np.array(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def test_manual_delivery_defaults_preserve_the_legacy_source() -> None:
    scenario = ImpactScenario(clubhead_speed_mph=30.0)
    config = SimulationConfig(scenario=scenario, club=_WEDGE)
    source = ManualSwingSource(scenario, delivery=config.manual_delivery)

    sample = source.sample(source.duration / 2.0)

    assert config.manual_attack_angle_deg == 0.0
    assert config.manual_club_path_deg == 0.0
    assert config.manual_forward_shaft_lean_deg == 0.0
    assert config.manual_shaft_axis_datum is ShaftAxisDatum.TRACKED_REFERENCE
    np.testing.assert_allclose(sample.pose[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(sample.twist[3:], (30.0 * 0.44704, 0.0, 0.0))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("manual_attack_angle_deg", math.nan),
        ("manual_attack_angle_deg", 90.0),
        ("manual_club_path_deg", -90.0),
        ("manual_forward_shaft_lean_deg", math.inf),
        ("manual_forward_shaft_lean_deg", 60.1),
    ),
)
def test_manual_delivery_rejects_nonfinite_or_singular_angles(
    field: str, value: float
) -> None:
    with pytest.raises(Exception, match=field.removeprefix("manual_")):
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=_WEDGE,
            **{field: value},
        )


def test_manual_delivery_rejects_unknown_shaft_datum() -> None:
    with pytest.raises(ValueError, match="shaft-axis datum"):
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=_WEDGE,
            manual_shaft_axis_datum="invented",  # type: ignore[arg-type]
        )


def test_manual_delivery_rejects_a_backward_facing_delivered_normal() -> None:
    with pytest.raises(Exception, match="delivered dynamic loft"):
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=_WEDGE,
            manual_forward_shaft_lean_deg=-50.0,
        )


def test_manual_source_applies_attack_path_pose_and_rotated_component_rates() -> None:
    scenario = ImpactScenario(
        clubhead_speed_mph=30.0,
        lie_angle_deg=64.0,
        omega_plane_dps=300.0,
        omega_shaft_dps=1307.0,
    )
    delivery = ManualDeliveryConfig(
        attack_angle_deg=-10.0,
        club_path_deg=5.0,
        forward_shaft_lean_deg=15.0,
    )
    source = ManualSwingSource(scenario, delivery=delivery)

    sample = source.sample(source.duration / 2.0)
    speed_mps = 30.0 * 0.44704
    attack = math.radians(-10.0)
    path = math.radians(5.0)
    expected_velocity = speed_mps * np.array(
        (
            math.cos(attack) * math.cos(path),
            math.sin(attack),
            math.cos(attack) * math.sin(path),
        )
    )
    expected_rotation = _forward_lean_rotation(15.0)
    expected_omega = expected_rotation @ np.radians(
        np.asarray(solve(scenario).omega_dps)
    )

    np.testing.assert_allclose(sample.twist[3:], expected_velocity, atol=1e-12)
    np.testing.assert_allclose(sample.pose[:3, :3], expected_rotation, atol=1e-12)
    np.testing.assert_allclose(sample.twist[:3], expected_omega, atol=1e-12)


def test_generated_hosel_datum_uses_declared_head_geometry_in_wedge_run() -> None:
    scenario = ImpactScenario(
        clubhead_speed_mph=30.0,
        lie_angle_deg=64.0,
        omega_plane_dps=0.0,
        omega_shaft_dps=1307.0,
        com_to_face_mm=20.0,
    )
    config = SimulationConfig(
        scenario=scenario,
        club=_WEDGE,
        impact_time_s=0.03,
        manual_attack_angle_deg=-10.0,
        manual_club_path_deg=0.0,
        manual_forward_shaft_lean_deg=15.0,
        manual_shaft_axis_datum=ShaftAxisDatum.GENERATED_HOSEL,
    )

    run = run_simulation(config)
    snapshot = impact_kinematics_for_run(run)
    rotation = _forward_lean_rotation(15.0)
    reference = np.asarray(snapshot.state.reference_position_m)
    local_hosel = np.asarray(hosel_point(_WEDGE))
    local_face_center = np.asarray(face_center_point(_WEDGE))
    registered_hosel = local_hosel - local_face_center
    registered_hosel[0] += scenario.com_to_face_mm / 1000.0
    local_shaft, _ = impact_frame(scenario.lie_angle_deg)

    assert snapshot.geometry_basis == "generated_head_profile_hosel"
    assert "representative" in snapshot.model_limitations
    np.testing.assert_allclose(
        snapshot.state.shaft_axis_point_m,
        reference + rotation @ registered_hosel,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        snapshot.state.shaft_axis_unit,
        rotation @ local_shaft,
        atol=1e-12,
    )
    assert run.delivery is not None
    delivery_velocity = np.asarray(run.delivery.clubhead_velocity)
    delivery_aoa = math.degrees(
        math.atan2(
            delivery_velocity[1], math.hypot(delivery_velocity[0], delivery_velocity[2])
        )
    )
    assert delivery_aoa == pytest.approx(-10.0)
    assert math.degrees(math.asin(run.delivery.face_normal[1])) == pytest.approx(
        _WEDGE.loft_deg - 15.0
    )
    assert snapshot.analysis.contact_velocity_mps == pytest.approx(
        (13.155691, -2.522013, -0.410056), abs=1e-6
    )
    assert snapshot.analysis.total_aoa_deg == pytest.approx(-10.847087, abs=1e-6)
    # Shaft-counterfactual decomposition repinned for #4799 G2: the
    # loft-aware wedge hosel moved the shaft axis to the leading edge,
    # changing the lever arm. Total delivery and carry are unchanged.
    assert snapshot.analysis.without_shaft_aoa_deg == pytest.approx(
        -11.270053, abs=1e-6
    )
    assert snapshot.analysis.shaft_counterfactual_aoa_delta_deg == pytest.approx(
        0.422965, abs=1e-6
    )
    assert snapshot.analysis.shaft_vertical_velocity_share == pytest.approx(
        0.018553, abs=1e-6
    )
    assert run.launch is not None
    assert run.launch["carry_m"] == pytest.approx(22.45855, abs=1e-5)

    contact_target_run = run_simulation(
        SimulationConfig(
            scenario=scenario,
            club=_WEDGE,
            impact_time_s=0.03,
            manual_attack_angle_deg=-9.1535118584,
            manual_forward_shaft_lean_deg=15.0,
            manual_shaft_axis_datum=ShaftAxisDatum.GENERATED_HOSEL,
        )
    )
    contact_target = impact_kinematics_for_run(contact_target_run).analysis
    assert contact_target.total_aoa_deg == pytest.approx(-10.0, abs=1e-9)
    # Repinned for #4799 G2 (see above): lever arm from the new hosel.
    assert contact_target.shaft_counterfactual_aoa_delta_deg == pytest.approx(
        0.373815, abs=1e-6
    )
    assert contact_target_run.launch is not None
    assert contact_target_run.launch["carry_m"] == pytest.approx(23.024061, abs=1e-6)


def test_run_document_persists_the_versioned_manual_delivery_contract() -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=_WEDGE,
            manual_attack_angle_deg=-10.0,
            manual_club_path_deg=4.0,
            manual_forward_shaft_lean_deg=15.0,
            manual_shaft_axis_datum=ShaftAxisDatum.GENERATED_HOSEL,
        )
    )

    document = run_to_json_dict(run)

    assert document["format"] == "rate_of_closure.simulation_run/5"
    assert document["parameters"]["manual_delivery"] == {
        "attack_angle_deg": -10.0,
        "club_path_deg": 4.0,
        "forward_shaft_lean_deg": 15.0,
        "shaft_axis_datum": "generated_hosel",
    }
    limitations = document["model_limitations"]
    assert limitations["contact_tracking"]["basis"] == "tracked_reference_point"
    assert limitations["impact_velocity"]["basis"] == "clubhead_reference_translation"
    assert manual_delivery_from_json_dict(document) == run.config.manual_delivery


def test_legacy_run_document_migrates_to_manual_delivery_defaults() -> None:
    legacy_document = {
        "format": "rate_of_closure.simulation_run/2",
        "parameters": {},
    }

    assert manual_delivery_from_json_dict(legacy_document) == ManualDeliveryConfig()
