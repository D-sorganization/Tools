"""Explicit derivation controls for shaft profile what-if studies."""

from __future__ import annotations

import pytest

from shared.python.golf_club import (
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftProfileScaling,
    ShaftStation,
    scale_shaft_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _profile() -> ShaftProfile:
    station = dict(
        outer_diameter_m=0.012,
        inner_diameter_m=0.010,
        linear_density_kg_m=0.08,
        ei_about_x_n_m2=40.0,
        ei_about_y_n_m2=32.0,
        gj_n_m2=20.0,
        damping_ratio=0.02,
        spine_angle_rad=0.1,
    )
    return ShaftProfile(
        shaft_id="baseline",
        frame_id="shaft",
        raw_length_m=1.0,
        cut_length_m=1.0,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=0.0,
        stations=(
            ShaftStation(position_m=0.0, **station),
            ShaftStation(position_m=1.0, **station),
        ),
        provenance=ShaftProfileProvenance(
            source_name="fixture",
            measurement_method="fixture",
            uncertainty_note="illustrative",
        ),
    )


def test_scaling_is_explicit_immutable_and_axis_specific() -> None:
    baseline = _profile()
    derived = scale_shaft_profile(
        baseline,
        ShaftProfileScaling(
            mass_scale=0.9,
            ei_about_x_scale=1.1,
            ei_about_y_scale=1.2,
            gj_scale=0.8,
            damping_scale=1.5,
        ),
        shaft_id="derived",
    )

    station = derived.stations[0]
    assert station.linear_density_kg_m == pytest.approx(0.072)
    assert station.ei_about_x_n_m2 == pytest.approx(44.0)
    assert station.ei_about_y_n_m2 == pytest.approx(38.4)
    assert station.gj_n_m2 == pytest.approx(16.0)
    assert station.damping_ratio == pytest.approx(0.03)
    assert station.outer_diameter_m == baseline.stations[0].outer_diameter_m
    assert baseline.stations[0].linear_density_kg_m == pytest.approx(0.08)
    assert derived.provenance.source_name == "derived from fixture"
    assert "scaling" in derived.provenance.measurement_method


@pytest.mark.parametrize("field", ShaftProfileScaling.__dataclass_fields__)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_scaling_rejects_nonpositive_or_nonfinite_factors(
    field: str, value: float
) -> None:
    values = {name: 1.0 for name in ShaftProfileScaling.__dataclass_fields__}
    values[field] = value

    with pytest.raises(ValueError, match=field):
        ShaftProfileScaling(**values)


def test_damping_scaling_cannot_create_nonphysical_ratio() -> None:
    with pytest.raises(ValueError, match="damping"):
        scale_shaft_profile(
            _profile(),
            ShaftProfileScaling(damping_scale=100.0),
            shaft_id="invalid",
        )
