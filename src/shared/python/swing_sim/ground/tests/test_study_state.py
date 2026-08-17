"""State-matrix and canonical-construction tests for ground studies."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from shared.python.swing_sim.ground import (
    GroundPhase,
    GroundResultStatus,
    GroundSummary,
    GroundTermination,
    GroundTerminationReason,
    GroundTrajectoryPoint,
    GroundUnavailableField,
    GroundUnavailableFieldId,
    GroundUnavailableReason,
)
from shared.python.swing_sim.ground.study_projection import project_ground_study
from shared.python.swing_sim.ground.study_record import GroundStudyProjection
from shared.python.swing_sim.ground.study_types import (
    GroundStudyStatus,
    GroundTargetUnavailableReason,
)
from shared.python.swing_sim.solver.spatial_targets import TargetPoint

from ._support import _failed_result, _request, _result
from .test_study_projection import _qualified_request, _target


def _partial_bounce_result(vertical_offset_m: float):
    source = _result()
    first = source.trajectory[0]
    final = GroundTrajectoryPoint(
        first.time_s + 0.1,
        first.frame,
        (
            first.position_m[0] + 1.0,
            first.position_m[1] + vertical_offset_m,
            first.position_m[2],
        ),
        (10.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        GroundPhase.BOUNCE,
    )
    summary = GroundSummary(
        math.hypot(first.position_m[0], first.position_m[2]),
        1.0,
        0.0,
        0.0,
        0.0,
        math.hypot(final.position_m[0], final.position_m[2]),
        final.position_m[0],
        final.position_m[2],
        0,
    )
    return replace(
        source,
        status=GroundResultStatus.PARTIAL,
        trajectory=(first, final),
        events=(source.events[0],),
        summary=summary,
        termination=GroundTermination(
            GroundTerminationReason.TIME_LIMIT,
            final.time_s,
            False,
        ),
        warnings=(),
    )


def test_wire_rejects_source_impossible_status_termination_pair() -> None:
    payload = project_ground_study(_qualified_request(), _result()).to_dict()
    payload.update(
        {
            "result_status": "failed",
            "status": "failed",
            "termination_reason": "rest",
            "metrics": None,
            "solver_eligibility": {
                "eligible": False,
                "reasons": ["result_not_complete", "missing_profile_binding"],
            },
        }
    )

    with pytest.raises(ValueError, match="incompatible"):
        GroundStudyProjection.from_dict(payload)


def test_wire_canonicalizes_target_numbers_before_returning_record() -> None:
    payload = project_ground_study(
        _qualified_request(),
        _result(),
        target=_target(),
    ).to_dict()
    target = payload["target"]
    assert isinstance(target, dict)
    position = target["position_m"]
    assert isinstance(position, dict)
    position["x"] = 228.000000000001

    parsed = GroundStudyProjection.from_dict(payload)

    assert parsed.target is not None and parsed.target.point.x_m == 228.0
    assert GroundStudyProjection.from_json(parsed.to_json()) == parsed


def test_direct_record_construction_canonicalizes_embedded_target() -> None:
    projection = project_ground_study(
        _qualified_request(),
        _result(),
        target=_target(),
    )
    assert projection.target is not None
    raw_target = replace(
        projection.target,
        point=TargetPoint(228.000000000001, 0.0, -2.25),
    )

    normalized = replace(projection, target=raw_target)

    assert normalized.target is not None and normalized.target.point.x_m == 228.0
    assert GroundStudyProjection.from_json(normalized.to_json()) == normalized


def test_partial_airborne_endpoint_is_censored_with_typed_target_unavailability() -> (
    None
):
    projection = project_ground_study(
        _request(),
        _partial_bounce_result(0.25),
        target=_target(),
    )

    assert projection.status is GroundStudyStatus.CENSORED
    assert projection.first_contact_target is not None
    assert projection.final_target is None
    assert (
        projection.final_target_unavailable_reason
        is GroundTargetUnavailableReason.ENDPOINT_AIRBORNE
    )
    assert GroundStudyProjection.from_json(projection.to_json()) == projection


def test_partial_penetrating_endpoint_is_rejected_not_called_airborne() -> None:
    with pytest.raises(ValueError, match="penetrates the bound plane"):
        project_ground_study(
            _request(),
            _partial_bounce_result(-0.25),
            target=_target(),
        )
    projection = project_ground_study(
        _request(),
        _partial_bounce_result(0.25),
        target=_target(),
    )
    assert projection.metrics is not None
    penetrating = replace(
        projection.metrics,
        final_observed_position_m=(211.0, -0.1, -3.0),
    )
    with pytest.raises(ValueError, match="penetrates the bound plane"):
        replace(projection, metrics=penetrating)

    payload = projection.to_dict()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics["final_observed_position_m"] = [211.0, -0.1, -3.0]
    with pytest.raises(ValueError, match="penetrates the bound plane"):
        GroundStudyProjection.from_dict(payload)


def test_targetless_record_revalidates_first_and_complete_final_contacts() -> None:
    projection = project_ground_study(_qualified_request(), _result())
    assert projection.metrics is not None
    forged_first = replace(
        projection.metrics,
        first_contact_position_m=(210.0, 10.02135, -3.0),
    )
    with pytest.raises(ValueError, match="does not contact the bound plane"):
        replace(projection, metrics=forged_first)

    payload = projection.to_dict()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics["final_observed_position_m"] = [228.0, 10.02135, -2.25]
    with pytest.raises(ValueError, match="does not contact the bound plane"):
        GroundStudyProjection.from_dict(payload)


def test_failed_and_unavailable_runs_never_fabricate_numeric_outputs() -> None:
    request = _qualified_request()
    failed = project_ground_study(request, _failed_result(), target=_target())
    assert failed.status is GroundStudyStatus.FAILED
    assert failed.metrics is None
    assert failed.first_contact_target is None
    assert failed.final_target is None

    unavailable_result = replace(
        _failed_result(),
        status=GroundResultStatus.UNAVAILABLE,
        unavailable_fields=(
            GroundUnavailableField(
                GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
                GroundUnavailableReason.NO_PHYSICAL_CONTACT,
                "flight event detector",
            ),
        ),
        termination=GroundTermination(
            GroundTerminationReason.UNAVAILABLE_INPUT,
            _failed_result().termination.time_s,
            False,
        ),
    )
    unavailable = project_ground_study(request, unavailable_result, target=_target())
    assert unavailable.status is GroundStudyStatus.UNAVAILABLE
    assert unavailable.metrics is None
    assert unavailable.unavailable_fields == unavailable_result.unavailable_fields
    assert GroundStudyProjection.from_json(unavailable.to_json()) == unavailable
