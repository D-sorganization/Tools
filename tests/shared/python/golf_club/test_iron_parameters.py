"""Contract tests for the versioned iron-head parameter schema (Tools #4149)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from shared.python.golf_club import (
    IRON_PARAMETERS_FORMAT,
    Handedness,
    IronPreset,
    iron_body_sections_m,
    iron_parameters_from_json,
    iron_parameters_to_json,
    iron_preset,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_presets_are_generic_and_progress_through_the_set() -> None:
    long_iron = iron_preset(IronPreset.LONG_IRON)
    mid_iron = iron_preset(IronPreset.MID_IRON)
    short_iron = iron_preset(IronPreset.SHORT_IRON)

    assert long_iron.loft_deg < mid_iron.loft_deg < short_iron.loft_deg
    assert long_iron.lie_deg < mid_iron.lie_deg < short_iron.lie_deg
    assert long_iron.target_mass_kg < mid_iron.target_mass_kg
    assert mid_iron.target_mass_kg < short_iron.target_mass_kg
    assert mid_iron.head_id == "generic-forged-iron-mid-iron"
    assert mid_iron.handedness is Handedness.RIGHT
    assert "not proprietary" in mid_iron.provenance.uncertainty_note


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("loft_deg", 15.0, r"loft_deg must be in \[16.0, 50.0\]"),
        ("lie_deg", 67.0, r"lie_deg must be in \[56.0, 66.0\]"),
        ("bounce_deg", -0.5, r"bounce_deg must be in \[0.0, 10.0\]"),
        ("blade_length_m", 0.095, r"blade_length_m must be in \[0.065, 0.09\]"),
        ("sole_width_m", 0.010, r"sole_width_m must be in \[0.012, 0.032\]"),
        ("offset_m", -0.001, r"offset_m must be in \[0.0, 0.008\]"),
        ("hosel_length_m", 0.10, r"hosel_length_m must be in \[0.04, 0.075\]"),
        ("target_mass_kg", float("nan"), "target_mass_kg must be finite"),
    ],
)
def test_out_of_range_fields_are_rejected_with_their_bounds(
    field: str, value: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(iron_preset(IronPreset.MID_IRON), **{field: value})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"heel_face_height_m": 0.046, "toe_face_height_m": 0.042},
            "toe_face_height_m must be at least heel_face_height_m",
        ),
        (
            {"hosel_outer_diameter_m": 0.012, "hosel_bore_diameter_m": 0.0100},
            "hosel wall must be at least 0.0015 m",
        ),
        (
            {"back_wall_height_m": 0.030, "heel_face_height_m": 0.030},
            "back_wall_height_m leaves less than 0.001 m below the heel topline",
        ),
        (
            {
                "loft_deg": 48.0,
                "heel_face_height_m": 0.045,
                "back_wall_height_m": 0.022,
                "sole_width_m": 0.012,
            },
            "back wall must sit at least 0.002 m behind the face plane",
        ),
    ],
)
def test_cross_field_geometry_is_rejected_at_the_schema_boundary(
    changes: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(iron_preset(IronPreset.MID_IRON), **changes)


def test_wrong_types_are_rejected() -> None:
    parameters = iron_preset(IronPreset.MID_IRON)
    with pytest.raises(TypeError, match="handedness"):
        dataclasses.replace(parameters, handedness="right")
    with pytest.raises(TypeError, match="loft_deg"):
        dataclasses.replace(parameters, loft_deg=True)
    with pytest.raises(TypeError, match="provenance"):
        dataclasses.replace(parameters, provenance=None)
    with pytest.raises(TypeError, match="preset"):
        iron_preset("mid_iron")  # type: ignore[arg-type]


def test_parameter_json_round_trips_and_is_versioned() -> None:
    parameters = iron_preset(IronPreset.LONG_IRON)

    text = iron_parameters_to_json(parameters)

    assert json.loads(text)["format"] == IRON_PARAMETERS_FORMAT
    assert IRON_PARAMETERS_FORMAT == "golf_club.iron_parameters/1"
    assert iron_parameters_from_json(text) == parameters
    assert iron_parameters_to_json(iron_parameters_from_json(text)) == text


def _set_format(document: dict[str, Any]) -> None:
    document["format"] = "golf_club.iron_parameters/0"


def _add_unknown_field(document: dict[str, Any]) -> None:
    document["parameters"]["extra"] = 1.0


def _bad_handedness(document: dict[str, Any]) -> None:
    document["parameters"]["handedness"] = "both"


def _bad_loft(document: dict[str, Any]) -> None:
    document["parameters"]["loft_deg"] = 80.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set_format, "unsupported iron-parameter format"),
        (_add_unknown_field, "extra"),
        (_bad_handedness, "unknown handedness"),
        (_bad_loft, "loft_deg"),
    ],
)
def test_parameter_json_rejects_foreign_or_invalid_documents(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    document = json.loads(iron_parameters_to_json(iron_preset(IronPreset.MID_IRON)))
    mutate(document)
    with pytest.raises(ValueError, match=message):
        iron_parameters_from_json(json.dumps(document))


def test_parameter_json_rejects_duplicate_fields() -> None:
    text = iron_parameters_to_json(iron_preset(IronPreset.MID_IRON))
    duplicated = text.replace('"format":', '"format":"x","format":', 1)
    with pytest.raises(ValueError, match="duplicate"):
        iron_parameters_from_json(duplicated)


def test_sections_place_toe_beyond_heel_with_declared_face_heights() -> None:
    parameters = iron_preset(IronPreset.MID_IRON)

    sections = iron_body_sections_m(parameters)

    assert sections.toe_z_m - sections.heel_z_m == pytest.approx(
        parameters.blade_length_m
    )
    heel = np.asarray(sections.heel_profile_m)
    toe = np.asarray(sections.toe_profile_m)
    assert float(np.linalg.norm(heel[1] - heel[0])) == pytest.approx(
        parameters.heel_face_height_m
    )
    assert float(np.linalg.norm(toe[1] - toe[0])) == pytest.approx(
        parameters.toe_face_height_m
    )
    assert float(min(heel[:, 1].min(), toe[:, 1].min())) == 0.0


def test_left_handed_sections_mirror_heel_and_toe() -> None:
    right = iron_preset(IronPreset.MID_IRON)
    left = dataclasses.replace(right, handedness=Handedness.LEFT)

    right_sections = iron_body_sections_m(right)
    left_sections = iron_body_sections_m(left)

    assert left_sections.heel_z_m == -right_sections.heel_z_m
    assert left_sections.toe_z_m == -right_sections.toe_z_m
    assert left_sections.heel_profile_m == right_sections.heel_profile_m
