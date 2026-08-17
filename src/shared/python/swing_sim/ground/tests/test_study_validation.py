"""Adversarial persistence and qualification tests for ground studies."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from shared.python.swing_sim.ground import (
    CalibrationKind,
    GroundCalibration,
    GroundWarning,
    GroundWarningSeverity,
)
from shared.python.swing_sim.ground.profile_binding import (
    ProfileOperatingCondition,
    SurfacePlacement,
    bind_material_profile,
)
from shared.python.swing_sim.ground.profile_types import (
    GroundEvidenceKind,
    GroundProfileRights,
)
from shared.python.swing_sim.ground.study_geometry import sphere_contact_point
from shared.python.swing_sim.ground.study_projection import project_ground_study
from shared.python.swing_sim.ground.study_record import GroundStudyProjection
from shared.python.swing_sim.ground.study_types import (
    GroundSolverEligibilityReason,
)
from shared.python.swing_sim.solver.spatial_targets import (
    SpatialTarget,
    SurfaceCircleTolerance,
    SurfaceCorridorTolerance,
    TargetPoint,
)

from ._support import _request, _result
from .test_profile_contract import _profile
from .test_study_projection import (
    _added,
    _bound_surface,
    _qualified_request,
    _scaled,
    _target,
    _tilted_case,
)


def test_intrinsic_target_boundary_is_accepted_and_miss_remains_solver_eligible() -> (
    None
):
    request, result, bound, downrange, final_contact = _tilted_case()
    on_edge_center = _added(final_contact, _scaled(downrange, -0.5))
    outside_center = _added(final_contact, _scaled(downrange, -0.500001))

    def target(center, tolerance):
        return SpatialTarget(
            "Intrinsic target",
            "landing_area",
            TargetPoint(*center),
            tolerance,
            "course_surface",
            bound.surface.surface_id,
        )

    on_edge = project_ground_study(
        request,
        result,
        bound_surface=bound,
        target=target(on_edge_center, SurfaceCircleTolerance(0.5)),
    )
    outside = project_ground_study(
        request,
        result,
        bound_surface=bound,
        target=target(
            outside_center,
            SurfaceCorridorTolerance(0.5, 0.25),
        ),
    )

    assert on_edge.final_target is not None and on_edge.final_target.accepted
    assert outside.final_target is not None and not outside.final_target.accepted
    assert outside.final_target.miss.distance_m == pytest.approx(0.000001)
    assert outside.solver_eligibility.eligible


def test_tilted_surface_contact_point_uses_ball_radius_along_plane_normal() -> None:
    center = (10.6, 4.8, -2.0)

    point = sphere_contact_point(center, (0.6, 0.8, 0.0), 1.0)

    assert point == pytest.approx((10.0, 4.0, -2.0))
    assert math.dist(center, point) == pytest.approx(1.0)


def test_tilted_surface_projection_uses_intrinsic_target_geometry() -> None:
    request, result, bound, _downrange, final_contact = _tilted_case()
    target = SpatialTarget(
        "Tilted final target",
        "landing_area",
        TargetPoint(*final_contact),
        SurfaceCircleTolerance(0.5),
        "course_surface",
        bound.surface.surface_id,
    )

    projection = project_ground_study(
        request,
        result,
        bound_surface=bound,
        target=target,
    )

    assert projection.final_target is not None
    assert projection.final_target.contact_point_m == pytest.approx(final_contact)
    assert projection.final_target.accepted
    assert projection.solver_eligibility.eligible
    assert projection.target is not None
    assert projection.target.point.x_m == 9.6
    assert projection.target.point.elevation_m == -7.2
    assert GroundStudyProjection.from_json(projection.to_json()) == projection


def test_target_center_must_lie_on_declared_surface_plane() -> None:
    request, result, bound, _downrange, final_contact = _tilted_case()
    off_plane = _added(final_contact, _scaled(bound.surface.normal_unit, 0.01))
    target = SpatialTarget(
        "Off-plane target",
        "landing_area",
        TargetPoint(*off_plane),
        SurfaceCircleTolerance(1.0),
        "course_surface",
        bound.surface.surface_id,
    )

    with pytest.raises(ValueError, match="target center"):
        project_ground_study(request, result, bound_surface=bound, target=target)


def test_profile_qualification_and_calibration_independently_gate_solver_use() -> None:
    unqualified = bind_material_profile(
        _profile(
            rights=GroundProfileRights(
                "LicenseRef-internal",
                "Fixture authors",
                False,
                False,
            )
        ),
        SurfacePlacement("firm-fairway", 0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        ProfileOperatingCondition("fairway", 290.0, 0.24),
    )
    illustrative = bind_material_profile(
        _profile(evidence_kind=GroundEvidenceKind.ENGINEERING_ESTIMATE),
        SurfacePlacement("firm-fairway", 0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        ProfileOperatingCondition("fairway", 290.0, 0.24),
    )

    unqualified_result = project_ground_study(
        replace(_request(), surface=unqualified.surface),
        _result(),
        bound_surface=unqualified,
    )
    illustrative_result = project_ground_study(
        replace(_request(), surface=illustrative.surface),
        _result(),
        bound_surface=illustrative,
    )

    assert unqualified_result.solver_eligibility.reasons == (
        GroundSolverEligibilityReason.PROFILE_UNQUALIFIED,
    )
    assert illustrative_result.solver_eligibility.reasons == (
        GroundSolverEligibilityReason.PROFILE_ILLUSTRATIVE,
    )


def test_unvalidated_result_calibration_blocks_solver_eligibility() -> None:
    calibration = GroundCalibration(
        "unvalidated-demo",
        CalibrationKind.UNVALIDATED,
        "no validation evidence",
        0.0,
    )
    request = replace(_qualified_request(), calibration=calibration)
    result = replace(_result(), calibration=calibration)

    projection = project_ground_study(
        request,
        result,
        bound_surface=_bound_surface(),
    )

    assert projection.calibration == calibration
    assert projection.provenance == result.provenance
    assert not projection.solver_eligibility.eligible
    assert projection.solver_eligibility.reasons == (
        GroundSolverEligibilityReason.MODEL_CALIBRATION_NOT_VALIDATED,
        GroundSolverEligibilityReason.MODEL_CALIBRATION_ZERO_CONFIDENCE,
    )
    assert GroundStudyProjection.from_json(projection.to_json()) == projection


def test_wire_cannot_forge_eligibility_after_downgrading_calibration() -> None:
    payload = project_ground_study(
        _qualified_request(),
        _result(),
        bound_surface=_bound_surface(),
    ).to_dict()
    calibration = payload["calibration"]
    assert isinstance(calibration, dict)
    calibration.update({"kind": "unvalidated", "confidence": 0.0})

    with pytest.raises(ValueError, match="solver eligibility"):
        GroundStudyProjection.from_dict(payload)


def test_target_evaluation_fails_closed_for_wrong_ground_source() -> None:
    with pytest.raises(ValueError, match="ground_source"):
        project_ground_study(
            _qualified_request(),
            _result(),
            bound_surface=_bound_surface(),
            target=_target(ground_source="different-ground"),
        )


def test_strict_study_parser_rejects_unknown_and_nonfinite_values() -> None:
    projection = project_ground_study(_qualified_request(), _result())
    payload = projection.to_dict()
    payload["invented"] = True
    with pytest.raises(ValueError, match="fields"):
        GroundStudyProjection.from_dict(payload)

    payload = projection.to_dict()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics["ground_elapsed_s"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        GroundStudyProjection.from_dict(payload)

    duplicate = projection.to_json().replace(
        '{"ball_radius_m":',
        '{"ball_radius_m":0.02135,"ball_radius_m":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate"):
        GroundStudyProjection.from_json(duplicate)


def test_study_parser_rejects_forged_solver_eligibility() -> None:
    projection = project_ground_study(_qualified_request(), _result())
    payload = projection.to_dict()
    payload["solver_eligibility"] = {
        "eligible": True,
        "reasons": [GroundSolverEligibilityReason.ELIGIBLE.value],
    }

    with pytest.raises(ValueError, match="solver eligibility"):
        GroundStudyProjection.from_dict(payload)


def test_wire_rejects_target_ground_source_mismatch() -> None:
    payload = project_ground_study(
        _qualified_request(),
        _result(),
        bound_surface=_bound_surface(),
        target=_target(),
    ).to_dict()
    target = payload["target"]
    assert isinstance(target, dict)
    target["ground_source"] = "different-ground"

    with pytest.raises(ValueError, match="ground_source"):
        GroundStudyProjection.from_dict(payload)


def test_wire_rejects_forged_target_miss() -> None:
    payload = project_ground_study(
        _qualified_request(),
        _result(),
        bound_surface=_bound_surface(),
        target=_target(),
    ).to_dict()
    evaluation = payload["first_contact_target"]
    assert isinstance(evaluation, dict)
    contact = evaluation["contact_point_m"]
    evaluation["miss"] = {
        "accepted": True,
        "closest_point_m": contact,
        "distance_m": 0.0,
        "vector_m": [0.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError, match="target miss"):
        GroundStudyProjection.from_dict(payload)


def test_wire_rejects_summary_endpoint_mismatch() -> None:
    payload = project_ground_study(_qualified_request(), _result()).to_dict()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    summary = metrics["summary"]
    assert isinstance(summary, dict)
    summary["final_offline_m"] = 42.0

    with pytest.raises(ValueError, match="final_offline"):
        GroundStudyProjection.from_dict(payload)


def test_wire_rejects_contact_point_not_equal_sphere_plane_contact() -> None:
    payload = project_ground_study(
        _qualified_request(),
        _result(),
        target=_target(),
    ).to_dict()
    evaluation = payload["first_contact_target"]
    assert isinstance(evaluation, dict)
    contact = evaluation["contact_point_m"]
    target_center = evaluation["target_center_m"]
    assert isinstance(contact, list) and isinstance(target_center, list)
    forged = [contact[0], contact[1] + 0.1, contact[2]]
    residual = [forged[index] - target_center[index] for index in range(3)]
    evaluation["contact_point_m"] = forged
    evaluation["center_residual_m"] = residual
    evaluation["center_distance_m"] = math.hypot(*residual)
    evaluation["miss"] = {
        "accepted": True,
        "closest_point_m": forged,
        "distance_m": 0.0,
        "vector_m": [0.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError, match="sphere-plane contact"):
        GroundStudyProjection.from_dict(payload)


def test_wire_rejects_profile_detached_from_solver_surface() -> None:
    payload = project_ground_study(
        _qualified_request(),
        _result(),
        bound_surface=_bound_surface(),
    ).to_dict()
    profile = payload["profile"]
    assert isinstance(profile, dict)
    material = profile["material_profile"]
    assert isinstance(material, dict)
    parameters = material["parameters"]
    assert isinstance(parameters, list)
    parameter = parameters[0]
    assert isinstance(parameter, dict)
    parameter["value_si"] = 0.43

    with pytest.raises(ValueError, match="surface material values"):
        GroundStudyProjection.from_dict(payload)


def test_typed_result_warning_round_trips_without_losing_evidence() -> None:
    warning = GroundWarning(
        "GROUND_STUDY_TEST_WARNING",
        GroundWarningSeverity.WARNING,
        "Fixture warning with retained severity and message.",
    )
    follow_up = GroundWarning(
        warning.code,
        GroundWarningSeverity.INFO,
        "A second record with the same code remains distinct.",
    )
    result = replace(_result(), warnings=(warning, follow_up))

    projection = project_ground_study(_qualified_request(), result)

    assert projection.warnings == (warning, follow_up)
    assert GroundStudyProjection.from_json(projection.to_json()) == projection
