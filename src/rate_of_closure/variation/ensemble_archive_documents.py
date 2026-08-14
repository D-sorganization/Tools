"""Strict metadata documents for the deterministic ensemble chunk archive."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from rate_of_closure.simulation.contact import ContactMode, ImpactOutcome, ImpactStatus
from shared.python.contracts import require

from .ensemble_trace_authority import (
    EnsembleAuthorityLayout,
    TrialContactEvent,
)
from .simulation_types import (
    ALL_OUTPUT_NAMES,
    SimulationTrialOutcome,
    TrialEvaluationStatus,
)


def exact_mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    """Return a string-keyed mapping with exactly the declared fields."""
    require(isinstance(value, Mapping), f"{name} must be an object")
    result = dict(cast(Mapping[object, object], value))
    require(all(isinstance(key, str) for key in result), f"{name} keys must be strings")
    keys = tuple(cast(str, key) for key in result)
    require(set(keys) == fields, f"{name} fields are invalid", sorted(keys))
    return cast(dict[str, Any], result)


def exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    """Return a genuine bounded integer."""
    require(type(value) is int and value >= minimum, f"{name} must be >= {minimum}")
    return cast(int, value)


def finite_number(value: object, name: str) -> float:
    """Return one real non-boolean finite JSON number."""
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{name} must be a finite number",
    )
    return float(cast(float, value))


def string_tuple(value: object, name: str) -> tuple[str, ...]:
    """Return an exact JSON string array."""
    require(isinstance(value, list), f"{name} must be an array")
    items = cast(list[object], value)
    require(
        all(isinstance(item, str) for item in items), f"{name} must contain strings"
    )
    return tuple(cast(list[str], items))


def outcome_document(outcome: SimulationTrialOutcome) -> dict[str, object]:
    """Encode one typed scalar outcome without inventing unavailable values."""
    return {
        "trial_index": outcome.trial_index,
        "status": outcome.status.value,
        "values": dict(outcome.values),
        "failure_type": outcome.failure_type,
        "failure_message": outcome.failure_message,
    }


def outcome_from_document(value: object) -> SimulationTrialOutcome:
    """Decode one exact typed scalar outcome."""
    data = exact_mapping(
        value,
        {"trial_index", "status", "values", "failure_type", "failure_message"},
        "outcome",
    )
    values = exact_mapping(data["values"], set(ALL_OUTPUT_NAMES), "outcome values")
    normalized: dict[str, float | None] = {}
    for name in ALL_OUTPUT_NAMES:
        item = values[name]
        normalized[name] = None if item is None else finite_number(item, name)
    failure_type = data["failure_type"]
    failure_message = data["failure_message"]
    require(
        failure_type is None or isinstance(failure_type, str), "invalid failure type"
    )
    require(
        failure_message is None or isinstance(failure_message, str),
        "invalid failure message",
    )
    require(isinstance(data["status"], str), "outcome status must be a string")
    try:
        status = TrialEvaluationStatus(data["status"])
    except ValueError as error:
        require(False, "unknown outcome status", data["status"])
        raise AssertionError from error
    return SimulationTrialOutcome(
        trial_index=exact_int(data["trial_index"], "trial_index"),
        status=status,
        values=normalized,
        failure_type=cast(str | None, failure_type),
        failure_message=cast(str | None, failure_message),
    )


def event_document(event: TrialContactEvent | None) -> dict[str, object] | None:
    """Encode a complete contact event or explicit failure absence."""
    if event is None:
        return None
    return {
        "trial_index": event.trial_index,
        "left_sample_index": event.left_sample_index,
        "right_sample_index": event.right_sample_index,
        "nearest_sample_index": event.nearest_sample_index,
        "outcome": event.outcome.to_dict(),
    }


def event_from_document(value: object) -> TrialContactEvent | None:
    """Decode one contact event with strict geometry metadata."""
    if value is None:
        return None
    data = exact_mapping(
        value,
        {
            "trial_index",
            "left_sample_index",
            "right_sample_index",
            "nearest_sample_index",
            "outcome",
        },
        "event",
    )
    raw = exact_mapping(
        data["outcome"],
        {
            "mode",
            "status",
            "candidate_time_s",
            "closest_approach_m",
            "contact_threshold_m",
            "contact_margin_m",
            "ball_position_m",
            "frame",
            "geometry_model",
            "geometry_limitations",
        },
        "contact outcome",
    )
    ball = raw["ball_position_m"]
    require(isinstance(ball, list) and len(ball) == 3, "ball position must be xyz")
    require(
        all(
            isinstance(raw[name], str)
            for name in (
                "mode",
                "status",
                "frame",
                "geometry_model",
                "geometry_limitations",
            )
        ),
        "contact identity fields must be strings",
    )
    ball_xyz = cast(
        tuple[float, float, float],
        tuple(
            finite_number(item, "ball position") for item in cast(list[object], ball)
        ),
    )
    outcome = ImpactOutcome(
        mode=ContactMode(raw["mode"]),
        status=ImpactStatus(raw["status"]),
        candidate_time_s=finite_number(raw["candidate_time_s"], "candidate_time_s"),
        closest_approach_m=finite_number(
            raw["closest_approach_m"], "closest_approach_m"
        ),
        contact_threshold_m=finite_number(
            raw["contact_threshold_m"], "contact_threshold_m"
        ),
        contact_margin_m=finite_number(raw["contact_margin_m"], "contact_margin_m"),
        ball_position_m=ball_xyz,
        frame=raw["frame"],
        geometry_model=raw["geometry_model"],
        geometry_limitations=raw["geometry_limitations"],
    )
    return TrialContactEvent(
        exact_int(data["trial_index"], "event trial_index"),
        outcome,
        exact_int(data["left_sample_index"], "left sample index"),
        exact_int(data["right_sample_index"], "right sample index"),
        exact_int(data["nearest_sample_index"], "nearest sample index"),
    )


def layout_document(layout: EnsembleAuthorityLayout) -> dict[str, object]:
    """Encode the exact supported trace authority layout."""
    return {
        "state_ids": list(layout.state_ids),
        "state_units": list(layout.state_units),
        "torque_joint_ids": list(layout.torque_joint_ids),
        "pose_frame": layout.pose_frame,
        "twist_component_ids": list(layout.twist_component_ids),
        "twist_units": list(layout.twist_units),
        "continuation_policy": layout.continuation_policy,
    }


def layout_from_document(value: object) -> EnsembleAuthorityLayout:
    """Decode the exact supported trace authority layout."""
    data = exact_mapping(
        value,
        {
            "state_ids",
            "state_units",
            "torque_joint_ids",
            "pose_frame",
            "twist_component_ids",
            "twist_units",
            "continuation_policy",
        },
        "authority layout",
    )
    for name in ("pose_frame", "continuation_policy"):
        require(isinstance(data[name], str), f"{name} must be a string")
    return EnsembleAuthorityLayout(
        state_ids=string_tuple(data["state_ids"], "state_ids"),
        state_units=string_tuple(data["state_units"], "state_units"),
        torque_joint_ids=string_tuple(data["torque_joint_ids"], "torque_joint_ids"),
        pose_frame=data["pose_frame"],
        twist_component_ids=string_tuple(
            data["twist_component_ids"], "twist_component_ids"
        ),
        twist_units=string_tuple(data["twist_units"], "twist_units"),
        continuation_policy=data["continuation_policy"],
    )


__all__ = [
    "event_document",
    "event_from_document",
    "exact_int",
    "exact_mapping",
    "finite_number",
    "layout_document",
    "layout_from_document",
    "outcome_document",
    "outcome_from_document",
    "string_tuple",
]
