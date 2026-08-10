"""Contract tests for the versioned flight-to-ground transfer boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground import (
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    GroundEventType,
    GroundPhase,
    GroundResultStatus,
    GroundSimulationRequest,
    GroundSimulationResult,
    GroundTermination,
    GroundTerminationReason,
    GroundUnavailableField,
    GroundUnavailableFieldId,
    GroundUnavailableReason,
    request_from_json,
    result_from_json,
)
from shared.python.swing_sim.ground.result_adapter import to_ground_model_result

from ._support import (
    _contact,
    _failed_result,
    _penetrating_contact,
    _request,
    _result,
    _surface,
)


def test_request_and_result_have_deterministic_strict_round_trips() -> None:
    request = _request()
    result = _result()

    assert request.schema_version == REQUEST_SCHEMA_VERSION
    assert result.schema_version == RESULT_SCHEMA_VERSION
    assert request_from_json(request.to_json()) == request
    assert result_from_json(result.to_json()) == result
    assert request_from_json(request.to_json()).to_json() == request.to_json()
    assert result_from_json(result.to_json()).to_json() == result.to_json()
    assert request.to_json() == canonical_numeric_json(request.to_dict())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(extra=True), "fields"),
        (lambda data: data.update(unit_system="imperial"), "unit_system"),
        (
            lambda data: data.update(schema_version="flight-to-ground-request/v2"),
            "schema_version",
        ),
        (lambda data: data["surface"].update(frame="world"), "(?i)frame"),
        (
            lambda data: data["surface"].update(normal_restitution=1.01),
            "normal_restitution",
        ),
        (
            lambda data: data["last_separated_state"].update(time_s=float("nan")),
            "time_s",
        ),
    ],
)
def test_request_parser_rejects_unknown_units_frames_ranges_and_nonfinite(
    mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    data = _request().to_dict()
    mutation(data)

    with pytest.raises((TypeError, ValueError), match=message):
        GroundSimulationRequest.from_dict(data)


def test_contract_invariants_reject_ambiguous_or_inconsistent_state() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        replace(_surface(), normal_unit=(0.0, 0.8, 0.0))
    with pytest.raises(ValueError, match="upward"):
        replace(_surface(), normal_unit=(0.0, -1.0, 0.0))
    with pytest.raises(ValueError, match="straddle"):
        replace(
            _request(),
            first_penetrating_state=replace(
                _penetrating_contact(), position_m=(210.0, 0.03, -3.0)
            ),
        )
    with pytest.raises(ValueError, match="output_interval_s"):
        replace(_request(), output_interval_s=13.0)
    with pytest.raises(ValueError, match="rotational_inertia_factor"):
        replace(_request(), rotational_inertia_factor=1.000000000001)
    with pytest.raises(ValueError, match="output_interval_s"):
        replace(_request(), max_time_s=1.0, output_interval_s=1.000000000004)


def test_result_rejects_bad_order_frames_event_sequences_and_status_matrix() -> None:
    result = _result()
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(result, trajectory=tuple(reversed(result.trajectory)))
    with pytest.raises(ValueError, match="event sequence"):
        replace(result, events=(replace(result.events[0], sequence=1),))
    with pytest.raises(ValueError, match="(?i)frame"):
        replace(
            result,
            trajectory=(replace(result.trajectory[0], frame=cast(Any, "other")),),
        )
    with pytest.raises(ValueError, match="completed"):
        replace(
            result,
            termination=GroundTermination(
                GroundTerminationReason.NUMERICAL_FAILURE, 8.0, True
            ),
        )


def test_result_rejects_inconsistent_summary_and_phase_state() -> None:
    result = _result()
    with pytest.raises(ValueError, match="bounce_count"):
        replace(result, summary=replace(result.summary, bounce_count=2))
    with pytest.raises(ValueError, match="surface path"):
        replace(result, summary=replace(result.summary, surface_path_distance_m=99.0))
    with pytest.raises(ValueError, match="rest phase"):
        replace(result.trajectory[-1], velocity_m_s=(0.1, 0.0, 0.0))
    regressed = replace(result.trajectory[2], phase=GroundPhase.BOUNCE)
    with pytest.raises(ValueError, match="phase transition"):
        replace(
            result, trajectory=(*result.trajectory[:2], regressed, result.trajectory[3])
        )


def test_result_rejects_inconsistent_event_state() -> None:
    result = _result()
    bad_first = replace(result.events[0], velocity_after_m_s=(99.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="output velocity"):
        replace(result, events=(bad_first, *result.events[1:]))
    duplicate_contact = replace(
        result.events[1], event_type=GroundEventType.FIRST_CONTACT
    )
    with pytest.raises(ValueError, match="event transition"):
        replace(
            result, events=(result.events[0], duplicate_contact, *result.events[2:])
        )
    bad_spin = replace(
        result.events[0],
        angular_velocity_after_rad_s=(1.0, 2.0, 3.0),
    )
    with pytest.raises(ValueError, match="output spin"):
        replace(result, events=(bad_spin, *result.events[1:]))


def test_result_rejects_inconsistent_terminal_state() -> None:
    result = _result()
    with pytest.raises(ValueError, match="zero output velocity"):
        replace(result.events[-1], velocity_after_m_s=(3.0, 4.0, 5.0))
    wrong_time = replace(result.events[-1], time_s=7.5)
    with pytest.raises(ValueError, match="terminal event time"):
        replace(result, events=(*result.events[:-1], wrong_time))
    wrong_position = replace(result.events[-1], position_m=(999.0, 0.0, 999.0))
    with pytest.raises(ValueError, match="terminal event position"):
        replace(result, events=(*result.events[:-1], wrong_position))
    with pytest.raises(ValueError, match="partial result cannot contain"):
        replace(
            result,
            status=GroundResultStatus.PARTIAL,
            termination=GroundTermination(
                GroundTerminationReason.TIME_LIMIT,
                result.termination.time_s,
                False,
            ),
        )


def test_result_parser_rejects_unknown_nested_fields_and_unsupported_version() -> None:
    nested = _result().to_dict()
    cast(dict[str, Any], nested["summary"])["invented"] = 1
    with pytest.raises(ValueError, match="fields"):
        GroundSimulationResult.from_dict(nested)

    unsupported = _result().to_dict()
    unsupported["schema_version"] = "flight-to-ground-result/v0"
    with pytest.raises(ValueError, match="schema_version"):
        GroundSimulationResult.from_dict(unsupported)


def test_failed_result_cannot_fabricate_trajectory_or_summary() -> None:
    failed = _failed_result()
    assert result_from_json(failed.to_json()) == failed

    with pytest.raises(ValueError, match="failed"):
        replace(failed, trajectory=_result().trajectory, summary=_result().summary)
    with pytest.raises(ValueError, match="incompatible"):
        replace(
            failed,
            termination=GroundTermination(
                GroundTerminationReason.UNAVAILABLE_INPUT,
                failed.termination.time_s,
                False,
            ),
        )


def test_unavailable_result_requires_typed_field_evidence() -> None:
    failed = _failed_result()
    unavailable = replace(
        failed,
        status=GroundResultStatus.UNAVAILABLE,
        unavailable_fields=(
            GroundUnavailableField(
                GroundUnavailableFieldId.TERMINAL_ANGULAR_VELOCITY,
                GroundUnavailableReason.SOURCE_DOES_NOT_PROPAGATE,
                "swing_sim.flight.models:waterloo_penner",
            ),
        ),
        termination=GroundTermination(
            GroundTerminationReason.UNAVAILABLE_INPUT,
            failed.termination.time_s,
            False,
        ),
    )
    with pytest.raises(ValueError, match="require unavailable_fields"):
        replace(
            failed,
            status=GroundResultStatus.UNAVAILABLE,
            termination=unavailable.termination,
        )
    with pytest.raises(ValueError, match="incompatible"):
        replace(
            unavailable,
            termination=GroundTermination(
                GroundTerminationReason.NUMERICAL_FAILURE,
                unavailable.termination.time_s,
                False,
            ),
        )


def test_contact_bracket_requires_an_incoming_relative_velocity() -> None:
    separating = replace(_contact(), velocity_m_s=(31.0, 12.0, 1.5))
    with pytest.raises(ValueError, match="incoming"):
        replace(_request(), last_separated_state=separating)

    stationary = replace(_contact(), velocity_m_s=(31.0, 0.0, 1.5))
    with pytest.raises(ValueError, match="incoming"):
        replace(
            _request(),
            last_separated_state=stationary,
            first_penetrating_state=replace(
                _penetrating_contact(), velocity_m_s=(31.0, 0.0, 1.5)
            ),
        )

    with pytest.raises(ValueError, match="tangential"):
        replace(_surface(), surface_velocity_m_s=(0.0, 1.0, 0.0))


def test_backward_adapter_maps_compatibility_values_without_inference() -> None:
    with pytest.deprecated_call(match="unqualified compatibility output"):
        legacy = to_ground_model_result(_result())

    assert legacy.model_id == "tools-ground-reference@0.1.0"
    assert legacy.total_distance_m == pytest.approx(228.0111017034039)
    assert legacy.roll_distance_m == pytest.approx(10.0)
    assert legacy.bounce_count == 1
    assert legacy.final_offline_m == pytest.approx(-2.25)
