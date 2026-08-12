from __future__ import annotations

import pytest

from shared.python.golf_club import (
    ExtrapolationPolicy,
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
)


def _station(position_m: float, *, linear_density_kg_m: float = 0.08) -> ShaftStation:
    return ShaftStation(
        position_m=position_m,
        outer_diameter_m=0.012,
        inner_diameter_m=0.010,
        linear_density_kg_m=linear_density_kg_m,
        ei_about_x_n_m2=42.0,
        ei_about_y_n_m2=38.0,
        gj_n_m2=24.0,
        damping_ratio=0.025,
        spine_angle_rad=0.1,
    )


def _profile() -> ShaftProfile:
    return ShaftProfile(
        shaft_id="measured-shaft",
        frame_id="shaft:butt_to_tip",
        raw_length_m=1.0,
        cut_length_m=0.9,
        tip_trim_m=0.04,
        butt_trim_m=0.06,
        insertion_depth_m=0.03,
        stations=(_station(0.0), _station(0.5), _station(1.0)),
        provenance=ShaftProfileProvenance(
            source_name="laboratory-bench-a",
            measurement_method="station-wise bending and torsion fixture",
            uncertainty_note="Values are illustrative test-fixture measurements.",
        ),
    )


def test_profile_is_immutable_and_mass_properties_are_integrated() -> None:
    profile = _profile()

    assert profile.total_mass_kg == pytest.approx(0.08)
    assert profile.cut_mass_kg == pytest.approx(0.072)
    assert profile.balance_point_from_raw_butt_m == pytest.approx(0.5)
    with pytest.raises(AttributeError):
        profile.raw_length_m = 2.0  # type: ignore[misc]


def test_profile_requires_ordered_endpoint_complete_stations() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ShaftProfile(
            shaft_id="bad-order",
            frame_id="shaft",
            raw_length_m=1.0,
            cut_length_m=1.0,
            tip_trim_m=0.0,
            butt_trim_m=0.0,
            insertion_depth_m=0.0,
            stations=(_station(0.0), _station(0.7), _station(0.6), _station(1.0)),
            provenance=ShaftProfileProvenance(
                source_name="fixture",
                measurement_method="fixture",
                uncertainty_note="fixture",
            ),
        )

    with pytest.raises(ValueError, match="raw butt and raw tip"):
        ShaftProfile(
            shaft_id="missing-tip",
            frame_id="shaft",
            raw_length_m=1.0,
            cut_length_m=1.0,
            tip_trim_m=0.0,
            butt_trim_m=0.0,
            insertion_depth_m=0.0,
            stations=(_station(0.0), _station(0.9)),
            provenance=ShaftProfileProvenance(
                source_name="fixture",
                measurement_method="fixture",
                uncertainty_note="fixture",
            ),
        )


def test_cut_length_must_equal_raw_length_less_declared_trims() -> None:
    with pytest.raises(ValueError, match="cut_length_m"):
        ShaftProfile(
            shaft_id="bad-trim",
            frame_id="shaft",
            raw_length_m=1.0,
            cut_length_m=0.95,
            tip_trim_m=0.04,
            butt_trim_m=0.06,
            insertion_depth_m=0.03,
            stations=(_station(0.0), _station(1.0)),
            provenance=ShaftProfileProvenance(
                source_name="fixture",
                measurement_method="fixture",
                uncertainty_note="fixture",
            ),
        )


def test_station_interpolation_is_explicit_and_rejects_extrapolation() -> None:
    profile = _profile()
    midpoint = profile.station_at(0.25)

    assert midpoint.position_m == pytest.approx(0.25)
    assert midpoint.ei_about_x_n_m2 == pytest.approx(42.0)
    assert midpoint.linear_density_kg_m == pytest.approx(0.08)
    with pytest.raises(ValueError, match="outside the measured station range"):
        profile.station_at(-0.01)

    clamped = profile.station_at(-0.01, ExtrapolationPolicy.CLAMP)
    assert clamped.position_m == pytest.approx(0.0)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("outer_diameter_m", 0.0, "outer_diameter_m"),
        ("inner_diameter_m", 0.012, "inner_diameter_m"),
        ("linear_density_kg_m", float("nan"), "linear_density_kg_m"),
        ("ei_about_x_n_m2", -1.0, "ei_about_x_n_m2"),
        ("damping_ratio", 1.0, "damping_ratio"),
    ],
)
def test_station_rejects_nonphysical_values(
    field: str,
    value: float,
    message: str,
) -> None:
    values = _station(0.0).__dict__ | {field: value}
    with pytest.raises(ValueError, match=message):
        ShaftStation(**values)
