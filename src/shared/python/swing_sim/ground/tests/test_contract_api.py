"""Public API and adversarial validation tests for the ground contract."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest

import shared.python.swing_sim.ground as ground
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from ._support import _request, _result, _surface

EXPECTED_API = {
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UNIT_SYSTEM_SI",
    "JSON_SCHEMA_DIALECT",
    "CANONICAL_GROUND_PARAMETER_IDS",
    "DEFAULT_PROFILE_LIBRARY_FILENAME",
    "DEFAULT_PROFILE_LIBRARY_MAX_BYTES",
    "GROUND_MATERIAL_PROFILE_SCHEMA_VERSION",
    "GROUND_PROFILE_LIBRARY_SCHEMA_VERSION",
    "GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION",
    "GROUND_STUDY_SCHEMA_VERSION",
    "PROFILE_UNQUALIFIED_WARNING",
    "PROFILE_ILLUSTRATIVE_WARNING",
    "UPSTREAM_TERRAIN_ADAPTER_VERSION",
    "UPSTREAM_TERRAIN_SNAPSHOT_SCHEMA_VERSION",
    "AdaptedTerrainSurface",
    "BoundGroundSurface",
    "CalibrationKind",
    "BounceModelSettings",
    "BounceAirSegment",
    "BounceTermination",
    "BounceTerminationReason",
    "GROUND_SKID_ROLL_MODEL_ID",
    "GROUND_SKID_ROLL_MODEL_VERSION",
    "MAX_REGIONAL_PLAN_REGIONS",
    "MAX_REGIONAL_PLAN_WIRE_BYTES",
    "MAX_REGIONAL_EXECUTION_STEPS",
    "MAX_REGIONAL_EXECUTION_TRANSITIONS",
    "MAX_REGIONAL_GROUND_EXECUTION_WIRE_BYTES",
    "MAX_REPEATED_BOUNCE_WIRE_BYTES",
    "MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES",
    "REGIONAL_PLAN_GEOMETRY_MODEL",
    "REGIONAL_PLAN_LIMITATIONS",
    "REGIONAL_PLAN_REQUEST_SCHEMA_VERSION",
    "REGIONAL_PLAN_RESULT_SCHEMA_VERSION",
    "REGIONAL_GROUND_EXECUTION_LIMITATIONS",
    "REGIONAL_GROUND_EXECUTION_SCHEMA_VERSION",
    "REGIONAL_GROUND_EXECUTOR_ID",
    "REGIONAL_GROUND_EXECUTOR_SOURCE",
    "REGIONAL_GROUND_EXECUTOR_VERSION",
    "REPEATED_BOUNCE_SCHEMA_VERSION",
    "REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION",
    "GroundCalibration",
    "GroundCalibrationRecord",
    "GroundApplicability",
    "GroundContactState",
    "GroundEvent",
    "GroundEventType",
    "GroundEndpointKind",
    "GroundFrame",
    "GroundEvidenceKind",
    "GroundMaterialParameter",
    "GroundMaterialProfile",
    "GroundModelUseStatus",
    "GroundParameterId",
    "GroundPhase",
    "GroundProvenance",
    "GroundRegionalMaterialPlanRequest",
    "GroundRegionalMaterialPlanResult",
    "GroundRegionalMaterialRegion",
    "RegionalGroundExecutionFailureReason",
    "RegionalGroundExecutionOptions",
    "RegionalGroundExecutionResult",
    "RegionalGroundExecutionStatus",
    "GroundProfileEvidence",
    "GroundProfileLibrary",
    "GroundProfileLibraryStore",
    "GroundProfileProvenance",
    "GroundProfileQualification",
    "GroundProfileRights",
    "GroundReferenceCancelled",
    "GroundReferenceExecution",
    "GroundReferenceExecutionError",
    "GroundReferencePhase",
    "GroundQualificationGate",
    "GroundQualificationGateId",
    "GroundQualificationStatus",
    "GroundResultStatus",
    "GroundSimulationRequest",
    "GroundSimulationResult",
    "GroundSolverEligibility",
    "GroundSolverEligibilityReason",
    "GroundStudyMetrics",
    "GroundStudyProfile",
    "GroundStudyProjection",
    "GroundStudyStatus",
    "GroundSummary",
    "GroundSurfaceProfile",
    "GroundTermination",
    "GroundTerminationReason",
    "GroundTargetEvaluation",
    "GroundTargetUnavailableReason",
    "GroundTrajectoryPoint",
    "GroundUnavailableField",
    "GroundUnavailableFieldId",
    "GroundUnavailableReason",
    "GroundWarning",
    "GroundWarningSeverity",
    "GroundCompositionError",
    "ImpactEnergyLedger",
    "ImpactImpulseResult",
    "ImpactRegime",
    "ImpactRejectionReason",
    "ImpactStateError",
    "FrameTransform",
    "SphereProperties",
    "RepeatedBounceResult",
    "RepeatedBounceRequest",
    "RepeatedBounceRequestResultPair",
    "PlanarSurfaceDomain",
    "PlanarSurfaceRegion",
    "ProfileOperatingCondition",
    "ProfileStoreConflictError",
    "ProfileStoreCorruptionError",
    "ProfileStoreError",
    "ProfileStoreIndeterminateCommitError",
    "ProfileStoreLockError",
    "ProfileStorePathError",
    "RigidMotion",
    "SkidRollEnergyLedger",
    "SkidRollExecution",
    "SkidRollResult",
    "SkidRollSettings",
    "SkidRollTermination",
    "SkidRollTerminationReason",
    "SurfaceBoundaryCrossing",
    "SurfaceKinematicSegment",
    "SurfaceRegionTransition",
    "SurfaceRegionTransitionCrossing",
    "SurfaceResolver",
    "SurfacePlacement",
    "StoredGroundProfileLibrary",
    "TerrainAdapterInterpretation",
    "TerrainDispositionKind",
    "TerrainFieldDisposition",
    "UpstreamTerrainSnapshot",
    "adapt_upstream_terrain_snapshot",
    "bind_material_profile",
    "compose_ground_result",
    "execute_repeated_bounce_request",
    "execute_regional_ground",
    "execution_input_sha256",
    "build_regional_material_plan_result",
    "interpolate_first_contact",
    "document_from_dict",
    "document_to_dict",
    "document_to_json",
    "library_from_json",
    "library_json_schema",
    "migrate_library_to_current",
    "migrate_profile_to_current",
    "profile_from_json",
    "profile_json_schema",
    "profile_schema_json",
    "project_ground_study",
    "qualified_study_to_ground_model_result",
    "request_from_json",
    "request_json_schema",
    "regional_material_plan_request_from_json",
    "regional_material_plan_result_from_json",
    "regional_plan_to_surface_resolver",
    "regional_ground_execution_result_from_json",
    "repeated_bounce_result_from_dict",
    "repeated_bounce_result_from_json",
    "repeated_bounce_result_to_dict",
    "repeated_bounce_result_to_json",
    "repeated_bounce_execution_input_sha256",
    "repeated_bounce_request_from_dict",
    "repeated_bounce_request_from_json",
    "repeated_bounce_request_to_dict",
    "repeated_bounce_request_to_json",
    "result_from_json",
    "result_json_schema",
    "schema_json",
    "migrate_request_to_current",
    "migrate_result_to_current",
    "resolve_sphere_plane_impact",
    "run_ground_reference",
    "simulate_repeated_bounce",
    "simulate_skid_roll",
    "upstream_snapshot_from_json",
    "validate_library_payload",
    "validate_profile_payload",
}


def test_public_api_is_explicit_and_package_is_self_facaded() -> None:
    assert set(ground.__all__) == EXPECTED_API


def test_ground_package_does_not_import_python_311_only_strenum() -> None:
    """Keep native StrEnum imports outside the Python 3.10 runtime path."""
    package_dir = Path(ground.__file__).parent
    offenders: list[str] = []
    for module_path in package_dir.glob("*.py"):
        syntax_tree = ast.parse(module_path.read_text(encoding="utf-8"))
        type_checking_nodes = {
            child
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
            for branch in node.body
            for child in ast.walk(branch)
        }
        for node in ast.walk(syntax_tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "enum":
                continue
            if node not in type_checking_nodes and any(
                alias.name == "StrEnum" for alias in node.names
            ):
                offenders.append(module_path.name)
    assert offenders == []


def test_ground_skid_roll_production_does_not_call_legacy_putting_integrator() -> None:
    package_dir = Path(ground.__file__).parent
    offenders: list[str] = []
    for module_path in package_dir.glob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        if "swing_sim.putting" in source or "from .putting" in source:
            offenders.append(module_path.name)

    assert offenders == []


@pytest.mark.parametrize(
    "change",
    [
        {"ball_radius_m": True},
        {"ball_mass_kg": False},
        {"rotational_inertia_factor": True},
        {"max_time_s": float("inf")},
        {"output_interval_s": float("-inf")},
        {"max_events": True},
    ],
)
def test_request_rejects_boolean_and_nonfinite_numeric_values(
    change: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        replace(_request(), **change)


def test_surface_rejects_material_range_errors() -> None:
    with pytest.raises(ValueError, match="kinetic_friction"):
        replace(_surface(), kinetic_friction=0.4)
    with pytest.raises(ValueError, match="moisture_fraction"):
        replace(_surface(), moisture_fraction=1.1)
    with pytest.raises(ValueError, match="turf_density_kg_m3"):
        replace(_surface(), turf_density_kg_m3=-1.0)
    with pytest.raises(ValueError, match="moisture_fraction"):
        replace(_surface(), moisture_fraction=-1e-12)
    with pytest.raises(ValueError, match="normal_restitution"):
        replace(_surface(), normal_restitution=1.000000000001)
    with pytest.raises(ValueError, match="kinetic_friction"):
        replace(_surface(), static_friction=0.3, kinetic_friction=0.300000000004)


def test_cross_runtime_integer_and_text_edges_fail_closed() -> None:
    unsafe_integer = 9_007_199_254_740_992
    with pytest.raises(ValueError, match="safe range"):
        replace(_request(), max_events=unsafe_integer)
    with pytest.raises(ValueError, match="safe range"):
        replace(_request(), max_events=10**1000)
    with pytest.raises(ValueError, match="safe range"):
        canonical_numeric_json(unsafe_integer)
    with pytest.raises(ValueError, match="safe range"):
        canonical_numeric_json(float(unsafe_integer))
    with pytest.raises(ValueError, match="safe range"):
        replace(_surface(), firmness_pa=float(unsafe_integer))
    with pytest.raises(ValueError, match="surrogate"):
        replace(_request(), request_id="\ud800")
    with pytest.raises(ValueError, match="surrogate"):
        canonical_numeric_json("\ud800")
    with pytest.raises(ValueError, match="at least"):
        replace(_request(), output_interval_s=1e-12)
    with pytest.raises(ValueError, match="whitespace"):
        replace(_request(), request_id=" ground-run-001 ")
    with pytest.raises(ValueError, match="whitespace"):
        replace(_request(), request_id="   ")


def test_records_are_frozen_and_vectors_are_immutable_tuples() -> None:
    request = _request()
    assert isinstance(request.last_separated_state.position_m, tuple)
    with pytest.raises(FrozenInstanceError):
        cast(Any, request).max_time_s = 2.0


def test_json_entry_points_fail_closed_for_nonobjects_and_invalid_json() -> None:
    with pytest.raises(ValueError, match="object"):
        ground.request_from_json("[]")
    with pytest.raises(ValueError, match="invalid"):
        ground.result_from_json("{")
    with pytest.raises(TypeError, match="text"):
        ground.request_from_json(cast(str, 2))
    duplicate_root = (
        _request()
        .to_json()
        .replace(
            '"request_id":"ground-run-001"',
            '"request_id":"first","request_id":"ground-run-001"',
            1,
        )
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        ground.request_from_json(duplicate_root)
    duplicate_nested = (
        _request()
        .to_json()
        .replace(
            '"surface_id":"firm-fairway"',
            '"surface_id":"first","surface_id":"firm-fairway"',
            1,
        )
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        ground.request_from_json(duplicate_nested)


def test_unqualified_compatibility_adapter_is_not_public() -> None:
    assert not hasattr(ground, "to_ground_model_result")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("surface", "bad"),
        ("calibration", "bad"),
        ("provenance", "bad"),
        ("last_separated_state", "bad"),
    ],
)
def test_request_direct_constructor_rejects_invalid_nested_records(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_request(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calibration", "bad"),
        ("provenance", "bad"),
        ("warnings", ("bad",)),
        ("unavailable_fields", ("bad",)),
        ("termination", "bad"),
    ],
)
def test_result_direct_constructor_rejects_invalid_nested_records(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        replace(_result(), **{field: value})
