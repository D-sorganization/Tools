"""Contract tests for the canonical modern-wedge parameter family."""

from __future__ import annotations

import pytest

from shared.python.golf_club import (
    Handedness,
    WedgeHeadParameters,
    WedgePreset,
    wedge_preset,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_presets_are_generic_distinct_and_driver_independent() -> None:
    low = wedge_preset(WedgePreset.LOW_BOUNCE)
    mid = wedge_preset(WedgePreset.MID_BOUNCE)
    high = wedge_preset(WedgePreset.HIGH_BOUNCE)

    assert (low.bounce_deg, mid.bounce_deg, high.bounce_deg) == (6.0, 10.0, 14.0)
    assert low.head_id == "generic-modern-wedge-low-bounce"
    assert high.handedness is Handedness.RIGHT
    assert all("generic" in item.head_id for item in (low, mid, high))
    assert low.provenance.source_name == "illustrative generic archetype"
    assert "not proprietary" in low.provenance.uncertainty_note


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("loft_deg", 70.0, "loft_deg"),
        ("lie_deg", 40.0, "lie_deg"),
        ("bounce_deg", -1.0, "bounce_deg"),
        ("sole_width_m", 0.05, "sole_width_m"),
        ("leading_edge_radius_m", 0.02, "leading_edge_radius_m"),
        ("target_mass_kg", float("nan"), "target_mass_kg"),
    ],
)
def test_parameters_reject_values_outside_supported_topology_domain(
    field: str, value: float, message: str
) -> None:
    values = wedge_preset(WedgePreset.MID_BOUNCE).__dict__ | {field: value}

    with pytest.raises(ValueError, match=message):
        WedgeHeadParameters(**values)


def test_hosel_bore_must_leave_a_minimum_wall() -> None:
    values = wedge_preset(WedgePreset.MID_BOUNCE).__dict__ | {
        "hosel_outer_diameter_m": 0.014,
        "hosel_bore_diameter_m": 0.013,
    }

    with pytest.raises(ValueError, match="hosel wall"):
        WedgeHeadParameters(**values)


def test_wrong_enum_contract_types_are_rejected() -> None:
    with pytest.raises(TypeError, match="handedness"):
        WedgeHeadParameters(
            **(wedge_preset(WedgePreset.MID_BOUNCE).__dict__ | {"handedness": "right"})
        )
    with pytest.raises(TypeError, match="preset"):
        wedge_preset("mid")  # type: ignore[arg-type]
