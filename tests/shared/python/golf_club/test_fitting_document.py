"""Round-trip and fail-closed gates for the OEM fitting document (C3, #4552)."""

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
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
)
from shared.python.golf_club.fitting_document import (
    FITTING_DOCUMENT_FORMAT,
    ClubFittingDocument,
    FaceGeometry,
    FittingProvenance,
    MeshReference,
    fitting_document_from_json,
    fitting_document_to_json,
)
from shared.python.golf_club.shaft_delivery import ShaftTipMass

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


def _shaft_profile() -> ShaftProfile:
    def station(position_m: float) -> ShaftStation:
        return ShaftStation(
            position_m=position_m,
            outer_diameter_m=0.012,
            inner_diameter_m=0.010,
            linear_density_kg_m=0.06,
            ei_about_x_n_m2=80.0,
            ei_about_y_n_m2=80.0,
            gj_n_m2=60.0,
            damping_ratio=0.025,
        )

    return ShaftProfile(
        shaft_id="fitting-doc-shaft",
        frame_id="shaft",
        raw_length_m=1.12,
        cut_length_m=1.12,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=0.0,
        stations=(station(0.0), station(1.12)),
        provenance=ShaftProfileProvenance(
            source_name="analytic fixture",
            measurement_method="uniform reference",
            uncertainty_note="exact synthetic values",
        ),
    )


def _document(*, with_mesh: bool = True) -> ClubFittingDocument:
    return ClubFittingDocument(
        document_id="fit-driver-2026-08-18",
        face=FaceGeometry(loft_deg=10.5, lie_deg=58.0, bulge_m=0.30, roll_m=0.28),
        assembly=_assembly(),
        shaft_profile=_shaft_profile(),
        tip_mass=ShaftTipMass(
            mass_kg=0.200, cg_back_m=0.012, cg_toe_m=0.030, cg_drop_m=0.040
        ),
        provenance=FittingProvenance(
            source_kind="oem_export",
            tool_name="acme-cad-exporter",
            exported_at="2026-08-18",
        ),
        mesh_reference=(
            MeshReference(name="driver-head.stl", sha256="ab" * 32, target_mass_kg=0.2)
            if with_mesh
            else None
        ),
    )


class TestRoundTrip:
    def test_serialization_is_deterministic_and_byte_exact(self) -> None:
        document = _document()
        first = fitting_document_to_json(document)
        second = fitting_document_to_json(document)
        assert first == second
        restored = fitting_document_from_json(first)
        assert fitting_document_to_json(restored) == first

    def test_round_trip_without_mesh_reference(self) -> None:
        document = _document(with_mesh=False)
        restored = fitting_document_from_json(fitting_document_to_json(document))
        assert restored.mesh_reference is None
        assert restored.face == document.face
        assert restored.tip_mass == document.tip_mass

    def test_wire_carries_the_versioned_format(self) -> None:
        payload = json.loads(fitting_document_to_json(_document()))
        assert payload["format"] == FITTING_DOCUMENT_FORMAT


class TestFailClosed:
    def test_unknown_top_level_field_is_refused(self) -> None:
        payload = json.loads(fitting_document_to_json(_document()))
        payload["surprise"] = 1
        with pytest.raises(ValueError, match="surprise|unknown"):
            fitting_document_from_json(json.dumps(payload))

    def test_unknown_nested_field_is_refused(self) -> None:
        payload = json.loads(fitting_document_to_json(_document()))
        payload["face"]["cor"] = 0.83
        with pytest.raises(ValueError, match="cor|unknown"):
            fitting_document_from_json(json.dumps(payload))

    def test_wrong_format_string_is_refused(self) -> None:
        payload = json.loads(fitting_document_to_json(_document()))
        payload["format"] = "golf_club.fitting_document/0"
        with pytest.raises(ValueError, match="format"):
            fitting_document_from_json(json.dumps(payload))

    def test_mesh_reference_requires_exactly_one_scale_selector(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            MeshReference(name="head.stl", sha256="ab" * 32)
        with pytest.raises(ValueError, match="exactly one"):
            MeshReference(
                name="head.stl",
                sha256="ab" * 32,
                density_kg_m3=4500.0,
                target_mass_kg=0.2,
            )

    def test_bad_sha_source_kind_and_date_are_refused(self) -> None:
        with pytest.raises(ValueError, match="sha256"):
            MeshReference(name="head.stl", sha256="nope", target_mass_kg=0.2)
        with pytest.raises(ValueError, match="source_kind"):
            FittingProvenance(
                source_kind="guessed", tool_name="tool", exported_at="2026-08-18"
            )
        with pytest.raises(ValueError, match="ISO-8601"):
            FittingProvenance(
                source_kind="measured", tool_name="tool", exported_at="today"
            )

    def test_face_geometry_bounds(self) -> None:
        with pytest.raises(ValueError, match="loft"):
            FaceGeometry(loft_deg=0.0, lie_deg=58.0)
        with pytest.raises(ValueError, match="lie"):
            FaceGeometry(loft_deg=10.5, lie_deg=20.0)
