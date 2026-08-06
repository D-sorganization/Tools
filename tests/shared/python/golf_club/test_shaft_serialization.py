"""Persistence contracts for measured shaft profiles."""

from __future__ import annotations

import json

import pytest

from shared.python.golf_club import (
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
    shaft_profile_from_csv,
    shaft_profile_from_json,
    shaft_profile_to_csv,
    shaft_profile_to_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _profile() -> ShaftProfile:
    provenance = ShaftProfileProvenance(
        source_name="laboratory-bench-a",
        measurement_method="three-point EI and torsion fixture",
        uncertainty_note="EI expanded uncertainty is 2 percent.",
        source_uri="https://example.invalid/shaft-a",
        data_license="CC-BY-4.0",
    )
    stations = tuple(
        ShaftStation(
            position_m=position,
            outer_diameter_m=outer,
            inner_diameter_m=inner,
            linear_density_kg_m=density,
            ei_about_x_n_m2=ei_x,
            ei_about_y_n_m2=ei_y,
            gj_n_m2=gj,
            damping_ratio=damping,
            spine_angle_rad=spine,
        )
        for position, outer, inner, density, ei_x, ei_y, gj, damping, spine in (
            (0.0, 0.0150, 0.0130, 0.070, 55.0, 50.0, 31.0, 0.022, 0.00),
            (0.5, 0.0125, 0.0108, 0.065, 42.0, 39.0, 25.0, 0.025, 0.08),
            (1.0, 0.0090, 0.0075, 0.060, 26.0, 24.0, 17.0, 0.030, 0.15),
        )
    )
    return ShaftProfile(
        shaft_id="measured-shaft-a",
        frame_id="shaft:butt_to_tip",
        raw_length_m=1.0,
        cut_length_m=0.91,
        tip_trim_m=0.03,
        butt_trim_m=0.06,
        insertion_depth_m=0.025,
        stations=stations,
        provenance=provenance,
    )


def test_json_round_trip_is_versioned_deterministic_and_lossless() -> None:
    profile = _profile()

    first = shaft_profile_to_json(profile)
    second = shaft_profile_to_json(profile)

    assert first == second
    assert json.loads(first)["format"] == "golf_club.shaft_profile/1"
    assert shaft_profile_from_json(first) == profile


def test_csv_round_trip_uses_explicit_si_headers_and_is_lossless() -> None:
    profile = _profile()

    text = shaft_profile_to_csv(profile)

    assert "position_m" in text.splitlines()[0]
    assert "ei_about_x_n_m2" in text.splitlines()[0]
    assert shaft_profile_from_csv(text) == profile


@pytest.mark.parametrize(
    ("loader", "payload", "error_type", "message"),
    [
        (shaft_profile_from_json, "not-json", ValueError, "valid JSON"),
        (
            shaft_profile_from_json,
            '{"format":"golf_club.shaft_profile/99"}',
            ValueError,
            "unsupported",
        ),
        (
            shaft_profile_from_json,
            '{"format":"golf_club.shaft_profile/1","format":"duplicate"}',
            ValueError,
            "duplicate field",
        ),
        (shaft_profile_from_csv, "unknown\n1\n", ValueError, "headers"),
    ],
)
def test_corrupt_unknown_or_ambiguous_documents_are_rejected(
    loader: object,
    payload: str,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        loader(payload)  # type: ignore[operator]


def test_json_unknown_fields_are_rejected() -> None:
    payload = json.loads(shaft_profile_to_json(_profile()))
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        shaft_profile_from_json(json.dumps(payload))


def test_csv_rejects_inconsistent_profile_metadata_between_rows() -> None:
    text = shaft_profile_to_csv(_profile())
    changed = text.replace("measured-shaft-a", "other-shaft", 1)

    with pytest.raises(ValueError, match="metadata"):
        shaft_profile_from_csv(changed)
