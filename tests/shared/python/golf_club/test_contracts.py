"""Contract tests for immutable golf-club domain records."""

from __future__ import annotations

import numpy as np
import pytest

import shared.python.golf_club as golf_club
from shared.python.golf_club import (
    AssembledMassProperties,
    ClubLengthConvention,
    ClubLengthMeasurement,
    ComponentMassProperties,
    ComponentRole,
    RigidTransform,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_public_facade_exports_the_stable_domain_api() -> None:
    assert set(golf_club.__all__) == {
        "CURRENT_FORMAT",
<<<<<<< HEAD
=======
        "CATEGORY_TURF",
>>>>>>> origin/feat/4271-regional-surface-transitions
        "LEGACY_FORMAT",
        "AssembledMassProperties",
        "ClubAssembly",
        "ClubComponent",
        "ClubLengthConvention",
        "ClubLengthMeasurement",
        "ComponentMassProperties",
        "ComponentRole",
<<<<<<< HEAD
        "RigidTransform",
        "assemble_mass_properties",
=======
        "ContactSequence",
        "ExtrapolationPolicy",
        "GroundPlane",
        "Handedness",
        "InstantaneousScrewAxis",
        "RigidTransform",
        "ReducedTurfContactResult",
        "SHAFT_PROFILE_FORMAT",
        "TURF_DAMPING_KEY",
        "TURF_FRICTION_KEY",
        "TURF_PENETRATION_LIMIT_KEY",
        "TURF_PROFILE_FORMAT",
        "TURF_STIFFNESS_KEY",
        "TURF_VARIABLE_KEYS",
        "WEDGE_EXPORT_FORMAT",
        "WEDGE_GROUND_CLEARANCE_FORMAT",
        "WEDGE_PARAMETERS_FORMAT",
        "ShaftModalResponse",
        "ShaftModalSettings",
        "ShaftProfile",
        "ShaftProfileProvenance",
        "ShaftProfileScaling",
        "ShaftStation",
        "ShaftTipLoad",
        "ShaftTipResponse",
        "TurfCalibrationStatus",
        "TurfContactKinematics",
        "TurfContactProfile",
        "TurfContactResponse",
        "TurfContactStatus",
        "TurfConvergenceStudy",
        "TurfPreset",
        "TurfProfileProvenance",
        "TurfVariationPlan",
        "WedgeExportArtifact",
        "WedgeExportFormat",
        "WedgeExportRequest",
        "WedgeExportResult",
        "WedgeClearanceSample",
        "WedgeContactCandidate",
        "WedgeContactFeature",
        "WedgeGeometryProvenance",
        "WedgeGroundClearanceAnalysis",
        "WedgeGroundContactEvent",
        "WedgeHeadParameters",
        "WedgeKinematicAnalysis",
        "WedgeKinematicState",
        "WedgeMeasuredMetrics",
        "WedgePreset",
        "WedgeSolidResult",
        "WedgeTurfPatchResponse",
        "WedgeTurfWrench",
        "assemble_mass_properties",
        "analyze_wedge_kinematics",
        "analyze_wedge_ground_clearance",
        "angle_of_attack_deg",
>>>>>>> origin/feat/4271-regional-surface-transitions
        "assembly_from_json",
        "assembly_from_json_dict",
        "assembly_to_json",
        "assembly_to_json_dict",
<<<<<<< HEAD
=======
        "build_wedge_solid",
        "export_wedge_artifacts",
        "evaluate_turf_contact",
        "evaluate_wedge_turf_wrench",
        "scale_shaft_profile",
        "shaft_component_mass_properties",
        "shaft_profile_from_csv",
        "shaft_profile_from_json",
        "shaft_profile_from_json_dict",
        "shaft_profile_to_csv",
        "shaft_profile_to_json",
        "shaft_profile_to_json_dict",
        "solve_cantilever_tip_response",
        "solve_shaft_bending_modes",
        "run_turf_convergence_study",
        "simulate_reduced_turf_contact",
        "turf_profile_from_json",
        "turf_profile_from_json_dict",
        "turf_profile_preset",
        "turf_profile_to_json",
        "turf_profile_to_json_dict",
        "turf_profiles_for_variation_plan",
        "wedge_parameters_from_json",
        "wedge_parameters_to_json",
        "wedge_body_profile_m",
        "wedge_contact_candidates",
        "wedge_face_contact_point_m",
        "wedge_ground_clearance_to_json_dict",
        "wedge_preset",
>>>>>>> origin/feat/4271-regional-surface-transitions
    }


def test_component_roles_are_stable_and_complete() -> None:
    assert tuple(role.value for role in ComponentRole) == (
        "head",
        "shaft",
        "grip",
        "adapter",
        "ferrule",
        "added_weight",
    )


def test_rigid_transform_maps_from_component_frame_to_club_frame() -> None:
    transform = RigidTransform(
        from_frame_id="head.frame",
        to_frame_id="club.datum",
        rotation=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        translation_m=(1.0, 2.0, 3.0),
    )

    assert transform.transform_point((2.0, 0.0, 0.0)) == pytest.approx((1.0, 4.0, 3.0))


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"from_frame_id": 2}, TypeError, "from_frame_id"),
        ({"translation_m": (0.0, 0.0)}, ValueError, "translation_m"),
        ({"translation_m": (0.0, np.inf, 0.0)}, ValueError, "finite"),
        (
            {"rotation": ((1.0, 0.0, 0.0),) * 3},
            ValueError,
            "proper orthonormal",
        ),
        (
            {"rotation": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))},
            ValueError,
            "proper orthonormal",
        ),
    ],
)
def test_rigid_transform_rejects_invalid_contracts(
    kwargs: dict[str, object], error_type: type[Exception], message: str
) -> None:
    values: dict[str, object] = {
        "from_frame_id": "component.frame",
        "to_frame_id": "club.frame",
    }
    values.update(kwargs)

    with pytest.raises(error_type, match=message):
        RigidTransform(**values)  # type: ignore[arg-type]


def test_mass_properties_defensively_copy_array_inputs() -> None:
    center = np.array([0.1, 0.2, 0.3])
    inertia = np.diag([1.0, 2.0, 3.0])
    properties = ComponentMassProperties(
        component_id="head-1",
        role=ComponentRole.HEAD,
        frame_id="head.frame",
        mass_kg=0.2,
        center_of_mass_m=center,
        inertia_at_com_kg_m2=inertia,
    )
    center[:] = 99.0
    inertia[:] = 99.0

    assert properties.center_of_mass_m == (0.1, 0.2, 0.3)
    assert properties.inertia_at_com_kg_m2 == (
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 3.0),
    )


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("component_id", "", ValueError, "component_id"),
        ("role", "head", TypeError, "role"),
        ("mass_kg", "0.2", TypeError, "mass_kg"),
        ("mass_kg", 0.0, ValueError, "mass_kg"),
        ("mass_kg", np.nan, ValueError, "mass_kg"),
        ("center_of_mass_m", (0.0, 0.0), ValueError, "center_of_mass_m"),
        (
            "inertia_at_com_kg_m2",
            ((1.0, 2.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ValueError,
            "symmetric",
        ),
        (
            "inertia_at_com_kg_m2",
            ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ValueError,
            "positive semidefinite",
        ),
        (
            "inertia_at_com_kg_m2",
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 3.0)),
            ValueError,
            "triangle inequality",
        ),
    ],
)
def test_mass_properties_reject_invalid_contracts(
    field: str, value: object, error_type: type[Exception], message: str
) -> None:
    values: dict[str, object] = {
        "component_id": "head-1",
        "role": ComponentRole.HEAD,
        "frame_id": "head.frame",
        "mass_kg": 0.2,
        "center_of_mass_m": (0.0, 0.0, 0.0),
        "inertia_at_com_kg_m2": np.diag([1.0, 1.0, 1.0]),
    }
    values[field] = value

    with pytest.raises(error_type, match=message):
        ComponentMassProperties(**values)  # type: ignore[arg-type]


def test_club_length_record_declares_reference_convention() -> None:
    measurement = ClubLengthMeasurement(
        length_m=1.143,
        convention=ClubLengthConvention.DECLARED_DATUMS,
        measurement_frame_id="club.measurement",
        lower_reference_id="sole-plane intersection",
        upper_reference_id="grip-cap end",
    )

    assert measurement.length_m == pytest.approx(1.143)
    assert measurement.lower_reference_id == "sole-plane intersection"


@pytest.mark.parametrize("length", [0.0, -1.0, np.nan, np.inf])
def test_club_length_must_be_finite_and_positive(length: float) -> None:
    with pytest.raises(ValueError, match="length_m"):
        ClubLengthMeasurement(
            length_m=length,
            convention=ClubLengthConvention.DECLARED_DATUMS,
            measurement_frame_id="club.measurement",
            lower_reference_id="lower",
            upper_reference_id="upper",
        )


def test_assembled_properties_reject_mutable_or_duplicate_component_ids() -> None:
    values = {
        "frame_id": "club.frame",
        "total_mass_kg": 1.0,
        "center_of_mass_m": (0.0, 0.0, 0.0),
        "inertia_at_com_kg_m2": np.eye(3),
    }
    with pytest.raises(TypeError, match="component_ids"):
        AssembledMassProperties(component_ids=["head"], **values)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        AssembledMassProperties(component_ids=("head", "head"), **values)
