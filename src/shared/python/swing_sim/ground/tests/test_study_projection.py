"""Qualified study projection tests for issue #4273."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import NamedTuple

import pytest

from shared.python.swing_sim.ground import (
    GroundContactState,
    GroundEvent,
    GroundEventType,
    GroundFrame,
    GroundPhase,
    GroundResultStatus,
    GroundSimulationRequest,
    GroundSimulationResult,
    GroundSummary,
    GroundTermination,
    GroundTerminationReason,
    GroundTrajectoryPoint,
)
from shared.python.swing_sim.ground.ground_result_composer import (
    compose_ground_result,
)
from shared.python.swing_sim.ground.profile_binding import (
    BoundGroundSurface,
    ProfileOperatingCondition,
    SurfacePlacement,
    bind_material_profile,
)
from shared.python.swing_sim.ground.skid_roll_simulation import simulate_skid_roll
from shared.python.swing_sim.ground.study_projection import project_ground_study
from shared.python.swing_sim.ground.study_record import GroundStudyProjection
from shared.python.swing_sim.ground.study_types import (
    GroundEndpointKind,
    GroundSolverEligibilityReason,
    GroundStudyStatus,
)
from shared.python.swing_sim.solver.spatial_targets import (
    SpatialTarget,
    SurfaceCircleTolerance,
    TargetPoint,
)

from ._support import (
    _request,
    _result,
    _settled_prefix,
    _surface,
    _surface_run_request,
)
from .test_profile_contract import _profile


def _bound_surface():
    return bind_material_profile(
        _profile(),
        SurfacePlacement(
            "firm-fairway",
            0.0,
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        ProfileOperatingCondition("fairway", 290.0, 0.24),
    )


def _qualified_request():
    return replace(_request(), surface=_bound_surface().surface)


def _target(*, ground_source: str = "firm-fairway") -> SpatialTarget:
    return SpatialTarget(
        label="Final rest target",
        kind="landing_area",
        point=TargetPoint(228.0, 0.0, -2.25),
        tolerance=SurfaceCircleTolerance(1.0),
        elevation_source="course_surface",
        ground_source=ground_source,
    )


def _scaled(vector: tuple[float, float, float], scale: float):
    return tuple(value * scale for value in vector)


def _added(*vectors: tuple[float, float, float]):
    return tuple(sum(vector[index] for vector in vectors) for index in range(3))


class _TiltedGeometry(NamedTuple):
    normal: tuple[float, float, float]
    downrange: tuple[float, float, float]
    first_contact: tuple[float, float, float]
    final_contact: tuple[float, float, float]
    first_center: tuple[float, float, float]
    roll_center: tuple[float, float, float]
    final_center: tuple[float, float, float]
    incoming: tuple[float, float, float]


def _tilted_geometry(
    normal: tuple[float, float, float],
    downrange: tuple[float, float, float],
    right: tuple[float, float, float],
    radius: float,
) -> _TiltedGeometry:
    first_contact = _scaled(downrange, 10.0)
    final_contact = _added(_scaled(downrange, 12.0), right)
    return _TiltedGeometry(
        normal,
        downrange,
        first_contact,
        final_contact,
        _added(first_contact, _scaled(normal, radius)),
        _added(_scaled(downrange, 11.0), _scaled(normal, radius)),
        _added(final_contact, _scaled(normal, radius)),
        _added(_scaled(downrange, 4.0), _scaled(normal, -0.2)),
    )


def _tilted_request(
    bound: BoundGroundSurface,
    geometry: _TiltedGeometry,
) -> GroundSimulationRequest:
    separated = GroundContactState(
        0.99,
        GroundFrame.TARGET,
        _added(geometry.first_center, _scaled(geometry.normal, 0.001)),
        geometry.incoming,
        (0.0, 0.0, 0.0),
    )
    penetrating = GroundContactState(
        1.01,
        GroundFrame.TARGET,
        _added(geometry.first_center, _scaled(geometry.normal, -0.001)),
        geometry.incoming,
        (0.0, 0.0, 0.0),
    )
    return replace(
        _request(),
        surface=bound.surface,
        last_separated_state=separated,
        first_penetrating_state=penetrating,
    )


def _tilted_points(geometry: _TiltedGeometry) -> tuple[GroundTrajectoryPoint, ...]:
    return (
        GroundTrajectoryPoint(
            1.0,
            GroundFrame.TARGET,
            geometry.first_center,
            _scaled(geometry.downrange, 3.0),
            (0.0, 0.0, 0.0),
            GroundPhase.IMPACT,
        ),
        GroundTrajectoryPoint(
            2.0,
            GroundFrame.TARGET,
            geometry.roll_center,
            _scaled(geometry.downrange, 1.0),
            (0.0, 0.0, 0.0),
            GroundPhase.ROLL,
        ),
        GroundTrajectoryPoint(
            3.0,
            GroundFrame.TARGET,
            geometry.final_center,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            GroundPhase.REST,
        ),
    )


def _tilted_events(
    geometry: _TiltedGeometry,
    points: tuple[GroundTrajectoryPoint, ...],
) -> tuple[GroundEvent, ...]:
    return (
        GroundEvent(
            0,
            GroundEventType.FIRST_CONTACT,
            1.0,
            GroundFrame.TARGET,
            geometry.first_center,
            geometry.incoming,
            points[0].velocity_m_s,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        GroundEvent(
            1,
            GroundEventType.SKID_TO_ROLL,
            2.0,
            GroundFrame.TARGET,
            geometry.roll_center,
            _scaled(geometry.downrange, 1.2),
            points[1].velocity_m_s,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        GroundEvent(
            2,
            GroundEventType.REST,
            3.0,
            GroundFrame.TARGET,
            geometry.final_center,
            _scaled(geometry.downrange, 0.1),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )


def _tilted_result(
    request: GroundSimulationRequest,
    bound: BoundGroundSurface,
    geometry: _TiltedGeometry,
) -> GroundSimulationResult:
    points = _tilted_points(geometry)
    summary = GroundSummary(
        math.hypot(geometry.first_center[0], geometry.first_center[2]),
        0.0,
        1.0,
        math.sqrt(2.0),
        1.0 + math.sqrt(2.0),
        math.hypot(geometry.final_center[0], geometry.final_center[2]),
        geometry.final_center[0],
        geometry.final_center[2],
        0,
    )
    return GroundSimulationResult(
        request.request_id,
        bound.surface.surface_id,
        GroundFrame.TARGET,
        "tilted-ground-fixture",
        "1.0.0",
        GroundResultStatus.COMPLETE,
        points,
        _tilted_events(geometry, points),
        summary,
        GroundTermination(GroundTerminationReason.REST, 3.0, True),
        request.calibration,
        (),
        (),
        request.provenance,
    )


def _tilted_case():
    normal = (0.6, 0.8, 0.0)
    downrange = (0.8, -0.6, 0.0)
    bound = bind_material_profile(
        _profile(),
        SurfacePlacement("tilted-fairway", 0.0, normal, (0.0, 0.0, 0.0)),
        ProfileOperatingCondition("fairway", 290.0, 0.24),
    )
    geometry = _tilted_geometry(
        normal,
        downrange,
        (0.0, 0.0, 1.0),
        _request().ball_radius_m,
    )
    request = _tilted_request(bound, geometry)
    return (
        request,
        _tilted_result(request, bound, geometry),
        bound,
        downrange,
        geometry.final_contact,
    )


def test_complete_rest_projection_preserves_metrics_target_and_profile() -> None:
    request = _qualified_request()
    projection = project_ground_study(
        request,
        _result(),
        bound_surface=_bound_surface(),
        target=_target(),
    )

    assert projection.status is GroundStudyStatus.COMPLETE
    assert (
        projection.result_sha256
        == hashlib.sha256(_result().to_json().encode("utf-8")).hexdigest()
    )
    assert projection.metrics is not None
    assert projection.metrics.summary == _result().summary
    assert projection.metrics.ground_elapsed_s == pytest.approx(2.8)
    assert projection.metrics.first_contact_position_m == (210.0, 0.02135, -3.0)
    assert projection.metrics.final_observed_position_m == (228.0, 0.02135, -2.25)
    assert projection.profile is not None
    assert projection.profile.profile_id == "fixture-fairway"
    assert projection.profile.profile_sha256 == _bound_surface().profile_sha256
    assert projection.solver_eligibility.eligible
    assert projection.solver_eligibility.reasons == (
        GroundSolverEligibilityReason.ELIGIBLE,
    )

    first = projection.first_contact_target
    final = projection.final_target
    assert first is not None and final is not None
    assert first.endpoint_kind is GroundEndpointKind.FIRST_CONTACT
    assert first.contact_point_m == (210.0, 0.0, -3.0)
    assert not first.accepted
    assert final.endpoint_kind is GroundEndpointKind.FINAL_REST
    assert final.ball_center_m == (228.0, 0.02135, -2.25)
    assert final.contact_point_m == (228.0, 0.0, -2.25)
    assert final.center_residual_m == (0.0, 0.0, 0.0)
    assert final.center_distance_m == 0.0
    assert final.accepted

    encoded = projection.to_json()
    assert encoded == projection.to_json()
    assert GroundStudyProjection.from_json(encoded) == projection


def test_projection_rejects_mismatched_request_result_and_surface_evidence() -> None:
    request = _qualified_request()
    bound = _bound_surface()

    with pytest.raises(ValueError, match="request_id"):
        project_ground_study(request, replace(_result(), request_id="other"))
    with pytest.raises(ValueError, match="surface"):
        project_ground_study(
            request,
            _result(),
            bound_surface=replace(
                bound,
                surface=replace(bound.surface, surface_id="different-surface"),
            ),
        )
    with pytest.raises(TypeError, match="exact GroundSimulationRequest"):
        project_ground_study(object(), _result())  # type: ignore[arg-type]
    points = _result().trajectory
    events = _result().events
    forged_position = (228.0, 0.12135, -2.25)
    off_plane = replace(
        _result(),
        trajectory=(
            *points[:-1],
            replace(points[-1], position_m=forged_position),
        ),
        events=(*events[:-1], replace(events[-1], position_m=forged_position)),
    )
    with pytest.raises(ValueError, match="does not contact the bound plane"):
        project_ground_study(request, off_plane)


def test_partial_run_is_censored_and_never_solver_eligible() -> None:
    surface = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=surface, max_time_s=0.2)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -2.0 / request.ball_radius_m),
    )
    result = compose_ground_result(request, prefix, simulate_skid_roll(request, prefix))
    projection = project_ground_study(request, result)

    assert result.status is GroundResultStatus.PARTIAL
    assert projection.status is GroundStudyStatus.CENSORED
    assert projection.metrics is not None
    assert projection.final_target is None
    assert not projection.solver_eligibility.eligible
    assert GroundSolverEligibilityReason.NOT_REST_TERMINATED in (
        projection.solver_eligibility.reasons
    )
