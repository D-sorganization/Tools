"""Strict selected-spec to shared ClubAssembly binding contract tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.club.assembly_binding import (
    CLUB_ASSEMBLY_BINDING_FORMAT,
    MAX_BINDING_BYTES,
    ClubAssemblySourceAuthority,
    MassPropertyAuthorityKind,
    build_club_assembly_binding,
    club_assembly_identity,
    club_assembly_identity_payload,
    club_spec_identity,
    club_spec_identity_payload,
    parse_club_assembly_binding,
    serialize_club_assembly_binding,
)
from rate_of_closure.club.engineering_sidecar import (
    build_clubhead_engineering_sidecar,
)
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

_DRIVER = "Driver 10.5\N{DEGREE SIGN}"
_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_FIXTURES = Path(__file__).parent / "fixtures"


def _component(
    component_id: str,
    role: ComponentRole,
    mass_kg: float,
    center_of_mass_m: tuple[float, float, float],
    inertia: tuple[tuple[float, float, float], ...],
    translation_m: tuple[float, float, float],
) -> ClubComponent:
    frame_id = f"{component_id}.frame"
    return ClubComponent(
        mass_properties=ComponentMassProperties(
            component_id=component_id,
            role=role,
            frame_id=frame_id,
            mass_kg=mass_kg,
            center_of_mass_m=center_of_mass_m,
            inertia_at_com_kg_m2=inertia,
        ),
        transform_to_club=RigidTransform(
            from_frame_id=frame_id,
            to_frame_id="driver.assembly",
            translation_m=translation_m,
        ),
    )


def _assembly(*, head_mass_kg: float = 0.2) -> ClubAssembly:
    return ClubAssembly(
        assembly_id="driver-qualified-2026-08",
        frame_id="driver.assembly",
        components=(
            _component(
                "head-qualified",
                ComponentRole.HEAD,
                head_mass_kg,
                (0.021, 0.026, -0.002),
                (
                    (1.4e-4, 3.0e-6, -2.0e-6),
                    (3.0e-6, 1.7e-4, 4.0e-6),
                    (-2.0e-6, 4.0e-6, 1.9e-4),
                ),
                (1.13, 0.0, 0.0),
            ),
            _component(
                "shaft-qualified",
                ComponentRole.SHAFT,
                0.075,
                (-0.45, 0.0, 0.0),
                (
                    (1.0e-6, 0.0, 0.0),
                    (0.0, 6.0e-3, 0.0),
                    (0.0, 0.0, 6.0e-3),
                ),
                (1.13, 0.0, 0.0),
            ),
            _component(
                "grip-qualified",
                ComponentRole.GRIP,
                0.05,
                (-1.06, 0.0, 0.0),
                (
                    (2.0e-6, 0.0, 0.0),
                    (0.0, 2.0e-4, 0.0),
                    (0.0, 0.0, 2.0e-4),
                ),
                (1.13, 0.0, 0.0),
            ),
        ),
        club_length=ClubLengthMeasurement(
            length_m=1.1557,
            convention=ClubLengthConvention.DECLARED_DATUMS,
            measurement_frame_id="driver.assembly",
            lower_reference_id="qualified sole-plane datum",
            upper_reference_id="qualified grip-cap datum",
        ),
    )


def _authority() -> ClubAssemblySourceAuthority:
    return ClubAssemblySourceAuthority(
        kind=MassPropertyAuthorityKind.QUALIFIED_ANALYSIS,
        authority_id="local-engineering-review",
        document_id="driver-mass-properties-contract-fixture",
        revision="1",
    )


def _binding():
    return build_club_assembly_binding(
        spec=get_club(_DRIVER),
        assembly=_assembly(),
        authority=_authority(),
        head_component_id="head-qualified",
        head_component_from_selected_head=RigidTransform(
            from_frame_id="rate_of_closure.head",
            to_frame_id="head-qualified.frame",
            rotation=_IDENTITY,
            translation_m=(0.001, -0.002, 0.003),
        ),
    )


def test_binding_round_trip_pins_both_identities_authority_units_and_frames() -> None:
    spec = get_club(_DRIVER)
    binding = _binding()
    payload = serialize_club_assembly_binding(binding)
    restored = parse_club_assembly_binding(spec, payload)
    document = json.loads(payload)

    assert restored == binding
    assert document["format"] == CLUB_ASSEMBLY_BINDING_FORMAT
    assert document["selected_spec_identity"]["sha256"] == club_spec_identity(spec)
    assert document["assembly_identity"] == {
        "assembly_id": _assembly().assembly_id,
        "format": "golf_club.assembly/1",
        "sha256": club_assembly_identity(_assembly()),
    }
    assert document["source_authority"]["kind"] == "qualified_analysis"
    assert document["units"] == {
        "angle": "degree",
        "inertia": "kg_m2",
        "length": "m",
        "mass": "kg",
    }
    assert (
        document["head_binding"]["head_component_from_selected_head"]["from_frame_id"]
        == "rate_of_closure.head"
    )


def test_binding_and_identity_bytes_match_cross_language_golden_fixtures() -> None:
    binding = _binding()
    binding_fixture = json.loads(
        (_FIXTURES / "club_assembly_binding_driver_10_5.json").read_text(
            encoding="utf-8"
        )
    )
    vectors = json.loads(
        (_FIXTURES / "club_assembly_binding_identity_vectors.json").read_text(
            encoding="utf-8"
        )
    )

    assert json.loads(serialize_club_assembly_binding(binding)) == binding_fixture
    assert (
        club_spec_identity_payload(binding.selected_spec).decode()
        == (vectors["selected_spec_identity_payload_utf8"])
    )
    assert binding.selected_spec_sha256 == vectors["selected_spec_sha256"]
    assert (
        club_assembly_identity_payload(binding.assembly).decode()
        == (vectors["assembly_identity_payload_utf8"])
    )
    assert binding.assembly_sha256 == vectors["assembly_sha256"]


def test_bound_sidecar_exposes_only_validated_head_and_assembly_properties() -> None:
    document = build_clubhead_engineering_sidecar(get_club(_DRIVER), _binding())
    head = document["mass_properties"]["head"]
    assembly = document["mass_properties"]["assembly"]

    assert document["capabilities"]["head_full_inertia_tensor"] == {
        "status": "available"
    }
    assert {
        key: value for key, value in head["center_of_mass_m"].items() if key != "value"
    } == {
        "frame_id": "rate_of_closure.head",
        "provenance": "validated_club_assembly_binding",
        "status": "available",
    }
    np.testing.assert_allclose(head["center_of_mass_m"]["value"], [0.02, 0.028, -0.005])
    np.testing.assert_allclose(
        head["inertia_tensor_at_com_kg_m2"]["value"],
        _assembly().components[0].mass_properties.inertia_at_com_kg_m2,
    )
    assert assembly["status"] == "available"
    assert assembly["frame_id"] == "driver.assembly"
    assert assembly["component_ids"] == [
        "head-qualified",
        "shaft-qualified",
        "grip-qualified",
    ]
    assert document["frames"]["world_from_head"]["status"] == "unavailable"
    assert document["provenance"]["assembly_binding"]["assembly_id"] == (
        "driver-qualified-2026-08"
    )


def test_binding_rejects_selected_spec_or_assembly_identity_mismatch() -> None:
    payload = serialize_club_assembly_binding(_binding())
    with pytest.raises(ValueError, match="selected ClubSpec identity"):
        parse_club_assembly_binding(replace(get_club(_DRIVER), loft_deg=11.0), payload)

    document = json.loads(payload)
    document["assembly"]["assembly_id"] = "substituted-assembly"
    with pytest.raises(ValueError, match="assembly identity"):
        parse_club_assembly_binding(get_club(_DRIVER), json.dumps(document))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document["source_authority"].update(
                {"kind": "representative_default"}
            ),
            "authority kind",
        ),
        (
            lambda document: document["head_binding"].update(
                {"head_component_id": "missing-head"}
            ),
            "head component",
        ),
        (
            lambda document: document["units"].update({"mass": "g"}),
            "units",
        ),
        (
            lambda document: document["head_binding"][
                "head_component_from_selected_head"
            ].update({"to_frame_id": "wrong.frame"}),
            "head component frame",
        ),
    ],
)
def test_binding_import_fails_closed_on_unqualified_or_inconsistent_data(
    mutate, message: str
) -> None:  # type: ignore[no-untyped-def]
    document = json.loads(serialize_club_assembly_binding(_binding()))
    mutate(document)
    with pytest.raises((TypeError, ValueError), match=message):
        parse_club_assembly_binding(get_club(_DRIVER), json.dumps(document))


def test_builder_rejects_head_mass_mismatch_and_missing_head_role() -> None:
    spec = get_club(_DRIVER)
    with pytest.raises(ValueError, match="head mass"):
        build_club_assembly_binding(
            spec=spec,
            assembly=_assembly(head_mass_kg=0.201),
            authority=_authority(),
            head_component_id="head-qualified",
            head_component_from_selected_head=RigidTransform(
                "rate_of_closure.head", "head-qualified.frame"
            ),
        )

    assembly = _assembly()
    no_head = replace(
        assembly,
        components=tuple(
            replace(
                component,
                mass_properties=replace(
                    component.mass_properties, role=ComponentRole.ADDED_WEIGHT
                ),
            )
            for component in assembly.components
        ),
    )
    with pytest.raises(ValueError, match="exactly one head"):
        build_club_assembly_binding(
            spec=spec,
            assembly=no_head,
            authority=_authority(),
            head_component_id="head-qualified",
            head_component_from_selected_head=RigidTransform(
                "rate_of_closure.head", "head-qualified.frame"
            ),
        )


def test_binding_import_rejects_duplicate_fields_and_oversize_documents() -> None:
    payload = serialize_club_assembly_binding(_binding()).decode("utf-8")
    ambiguous = payload.replace(
        '{\n  "assembly":',
        '{\n  "format": "rate_of_closure.club_assembly_binding/1",\n  "assembly":',
        1,
    )
    assert ambiguous != payload

    with pytest.raises(ValueError, match="duplicate field 'format'"):
        parse_club_assembly_binding(get_club(_DRIVER), ambiguous)

    with pytest.raises(ValueError, match="exceeds the 4 MiB limit"):
        parse_club_assembly_binding(get_club(_DRIVER), " " * (MAX_BINDING_BYTES + 1))
