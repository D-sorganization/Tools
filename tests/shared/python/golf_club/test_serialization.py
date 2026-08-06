"""Versioned deterministic JSON tests for golf-club assemblies."""

from __future__ import annotations

import json

import pytest

from shared.python.golf_club import (
    ClubAssembly,
    ClubComponent,
    ClubLengthConvention,
    ClubLengthMeasurement,
    ComponentMassProperties,
    ComponentRole,
    RigidTransform,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _assembly() -> ClubAssembly:
    properties = ComponentMassProperties(
        component_id="head-1",
        role=ComponentRole.HEAD,
        frame_id="head.frame",
        mass_kg=0.2,
        center_of_mass_m=(0.01, 0.0, 0.0),
        inertia_at_com_kg_m2=(
            (0.001, 0.0, 0.0),
            (0.0, 0.002, 0.0),
            (0.0, 0.0, 0.003),
        ),
    )
    return ClubAssembly(
        assembly_id="driver-demo",
        frame_id="club.frame",
        components=(
            ClubComponent(
                properties,
                RigidTransform(
                    from_frame_id="head.frame",
                    to_frame_id="club.frame",
                    translation_m=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        club_length=ClubLengthMeasurement(
            length_m=1.143,
            convention=ClubLengthConvention.DECLARED_DATUMS,
            measurement_frame_id="club.frame",
            lower_reference_id="sole-plane intersection",
            upper_reference_id="grip-cap end",
        ),
    )


def test_json_round_trip_is_versioned_deterministic_and_lossless() -> None:
    assembly = _assembly()

    first = assembly.to_json()
    second = assembly.to_json()
    restored = ClubAssembly.from_json(first)

    assert first == second
    assert json.loads(first)["format"] == "golf_club.assembly/1"
    assert restored == assembly
    assert restored.mass_properties == assembly.mass_properties


def test_version_zero_migrates_declared_length_without_changing_geometry() -> None:
    legacy = _assembly().to_json_dict()
    legacy["format"] = "golf_club.assembly/0"
    measurement = legacy.pop("club_length")
    legacy["club_length_m"] = measurement["length_m"]

    restored = ClubAssembly.from_json_dict(legacy)

    assert restored.club_length == ClubLengthMeasurement(
        length_m=1.143,
        convention=ClubLengthConvention.DECLARED_DATUMS,
        measurement_frame_id="club.frame",
        lower_reference_id="unspecified legacy lower datum",
        upper_reference_id="unspecified legacy upper datum",
    )


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        ("not-json", ValueError, "valid JSON"),
        ("[]", TypeError, "object"),
        (
            '{"format":"golf_club.assembly/99"}',
            ValueError,
            "unsupported",
        ),
    ],
)
def test_corrupt_or_unknown_json_is_rejected(
    payload: str, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        ClubAssembly.from_json(payload)


def test_unknown_fields_are_rejected_instead_of_silently_discarded() -> None:
    payload = _assembly().to_json_dict()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        ClubAssembly.from_json_dict(payload)


def test_duplicate_json_fields_are_rejected_as_ambiguous() -> None:
    payload = (
        _assembly()
        .to_json()
        .replace(
            '"assembly_id":"driver-demo"',
            '"assembly_id":"driver-demo","assembly_id":"other"',
        )
    )

    with pytest.raises(ValueError, match="duplicate field 'assembly_id'"):
        ClubAssembly.from_json(payload)


def test_json_loader_rejects_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="text"):
        ClubAssembly.from_json(4)  # type: ignore[arg-type]
