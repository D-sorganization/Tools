"""Shaft-to-assembly mass-property adapter tests."""

from __future__ import annotations

import pytest

from shared.python.golf_club import (
    ComponentRole,
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
    shaft_component_mass_properties,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _uniform_profile() -> ShaftProfile:
    values = dict(
        outer_diameter_m=0.012,
        inner_diameter_m=0.010,
        linear_density_kg_m=0.08,
        ei_about_x_n_m2=40.0,
        ei_about_y_n_m2=32.0,
        gj_n_m2=20.0,
        damping_ratio=0.02,
    )
    return ShaftProfile(
        shaft_id="uniform-shaft",
        frame_id="shaft-local-z",
        raw_length_m=1.0,
        cut_length_m=1.0,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=0.03,
        stations=(
            ShaftStation(position_m=0.0, **values),
            ShaftStation(position_m=1.0, **values),
        ),
        provenance=ShaftProfileProvenance(
            source_name="analytic fixture",
            measurement_method="uniform reference",
            uncertainty_note="exact synthetic values",
        ),
    )


def test_uniform_shaft_maps_to_full_three_dimensional_mass_properties() -> None:
    profile = _uniform_profile()

    properties = shaft_component_mass_properties(profile)

    mass = 0.08
    radial_squared = 0.012**2 + 0.010**2
    expected_transverse = mass / 12.0 + mass * radial_squared / 16.0
    expected_polar = mass * radial_squared / 8.0
    assert properties.component_id == "uniform-shaft"
    assert properties.role is ComponentRole.SHAFT
    assert properties.frame_id == "shaft-local-z"
    assert properties.mass_kg == pytest.approx(mass)
    assert properties.center_of_mass_m == pytest.approx((0.0, 0.0, 0.5))
    assert properties.inertia_at_com_kg_m2[0][0] == pytest.approx(
        expected_transverse, rel=1e-10
    )
    assert properties.inertia_at_com_kg_m2[1][1] == pytest.approx(
        expected_transverse, rel=1e-10
    )
    assert properties.inertia_at_com_kg_m2[2][2] == pytest.approx(
        expected_polar, rel=1e-10
    )


def test_adapter_rejects_wrong_contract_type() -> None:
    with pytest.raises(TypeError, match="profile"):
        shaft_component_mass_properties(object())  # type: ignore[arg-type]
