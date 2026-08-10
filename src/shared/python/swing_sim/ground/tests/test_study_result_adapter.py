"""Qualified study projection into the existing flight ground-metric DTO."""

from dataclasses import replace

import pytest

from shared.python.swing_sim.ground import (
    CalibrationKind,
    GroundCalibration,
    qualified_study_to_ground_model_result,
)
from shared.python.swing_sim.ground.study_projection import project_ground_study
from shared.python.swing_sim.ground.study_record import GroundStudyProjection
from shared.python.swing_sim.solver.spatial_targets import SpatialTarget, TargetPoint

from ._support import _result
from .test_study_projection import _bound_surface, _qualified_request, _target


def _qualified_study(
    *, target: SpatialTarget | None = None
) -> GroundStudyProjection:
    return project_ground_study(
        _qualified_request(),
        _result(),
        bound_surface=_bound_surface(),
        target=target,
    )


def test_qualified_study_populates_existing_ground_model_result() -> None:
    output = qualified_study_to_ground_model_result(_qualified_study())

    assert output.model_id == "tools-ground-reference@0.1.0"
    assert output.total_distance_m == pytest.approx(228.0111017034039)
    assert output.roll_distance_m == pytest.approx(10.0)
    assert output.bounce_count == 1
    assert output.final_offline_m == pytest.approx(-2.25)


def test_target_miss_does_not_hide_qualified_physics_metrics() -> None:
    missed_target = replace(_target(), point=TargetPoint(500.0, 0.0, 0.0))
    study = _qualified_study(target=missed_target)
    assert study.final_target is not None and not study.final_target.accepted

    output = qualified_study_to_ground_model_result(study)

    assert study.metrics is not None
    assert output.total_distance_m == study.metrics.summary.total_distance_m


def test_ineligible_or_wrong_type_study_fails_closed() -> None:
    unbound = project_ground_study(_qualified_request(), _result())
    with pytest.raises(ValueError, match="solver-eligible"):
        qualified_study_to_ground_model_result(unbound)
    with pytest.raises(TypeError, match="exact GroundStudyProjection"):
        qualified_study_to_ground_model_result(object())  # type: ignore[arg-type]


def test_unvalidated_result_model_cannot_enter_ground_metric_dto() -> None:
    calibration = GroundCalibration(
        "unvalidated-demo",
        CalibrationKind.UNVALIDATED,
        "no validation evidence",
        0.0,
    )
    request = replace(_qualified_request(), calibration=calibration)
    result = replace(_result(), calibration=calibration)
    study = project_ground_study(
        request,
        result,
        bound_surface=_bound_surface(),
    )

    with pytest.raises(ValueError, match="solver-eligible"):
        qualified_study_to_ground_model_result(study)
