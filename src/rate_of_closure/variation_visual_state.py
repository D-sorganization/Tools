"""Strict, UI-neutral visual-state authority for Variation evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from types import MappingProxyType

import numpy as np

from rate_of_closure.simulation import SimulationConfig
from shared.python.swing_sim.variation import VariationPlan

_FIXTURE = (
    Path(__file__).parent
    / "web/src/model/__fixtures__/variation_visual_state_matrix_v1.json"
)
_ROOT_KEYS = {"schema_id", "schema_version", "states"}
_ROW_KEYS = {"event", "phase", "visual_origin", "announcement_role"}


class VariationVisualEvent(StrEnum):
    INVALIDATE = "invalidate"
    START_EMPTY = "start-empty"
    START_RETAINED = "start-retained"
    SUCCEED = "succeed"
    FAIL_EMPTY = "fail-empty"
    FAIL_RETAINED = "fail-retained"
    CANCEL_EMPTY = "cancel-empty"
    CANCEL_RETAINED = "cancel-retained"


class VariationVisualPhase(StrEnum):
    EMPTY = "empty"
    LOADING = "loading"
    RESULT = "result"
    ERROR = "error"


class VariationVisualOrigin(StrEnum):
    EMPTY_PREVIEW = "empty-preview"
    PRIOR_ACCEPTED = "prior-accepted"
    CURRENT_ACCEPTED = "current-accepted"


class AnnouncementRole(StrEnum):
    STATUS = "status"
    ALERT = "alert"


@dataclass(frozen=True)
class VariationVisualState:
    phase: VariationVisualPhase
    visual_origin: VariationVisualOrigin
    announcement_role: AnnouncementRole


def _exact(value: object, keys: set[str], context: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise TypeError(f"{context} must contain exact fields {sorted(keys)}")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def parse_variation_visual_state_matrix(
    document: object,
) -> Mapping[VariationVisualEvent, VariationVisualState]:
    """Parse the exact shared transition matrix and reject semantic gaps."""
    root = _exact(document, _ROOT_KEYS, "visual state matrix")
    if root["schema_id"] != "rate-of-closure/variation-visual-state-matrix":
        raise ValueError("unsupported visual state matrix schema_id")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValueError("unsupported visual state matrix schema_version")
    rows = root["states"]
    if not isinstance(rows, list):
        raise TypeError("visual state matrix states must be a list")
    result: dict[VariationVisualEvent, VariationVisualState] = {}
    for raw in rows:
        row = _exact(raw, _ROW_KEYS, "visual state row")
        event = VariationVisualEvent(_string(row["event"], "visual state event"))
        if event in result:
            raise ValueError(f"duplicate visual state event: {event.value}")
        result[event] = VariationVisualState(
            VariationVisualPhase(_string(row["phase"], "visual state phase")),
            VariationVisualOrigin(_string(row["visual_origin"], "visual origin")),
            AnnouncementRole(_string(row["announcement_role"], "announcement role")),
        )
    if set(result) != set(VariationVisualEvent):
        raise ValueError("visual state matrix must define every event exactly once")
    return MappingProxyType(result)


_MATRIX = parse_variation_visual_state_matrix(
    json.loads(_FIXTURE.read_text(encoding="utf-8"))
)


def variation_visual_state(event: VariationVisualEvent) -> VariationVisualState:
    """Resolve one immutable visual state from the shared matrix."""
    if not isinstance(event, VariationVisualEvent):
        raise TypeError("event must be a VariationVisualEvent")
    return _MATRIX[event]


def _authority_value(value: object) -> object:
    """Freeze a complete immutable authority graph without lossy coercion."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return ("float", value.hex())
    if isinstance(value, Enum):
        return (type(value).__qualname__, _authority_value(value.value))
    if isinstance(value, np.ndarray):
        flat_values: list[object] = []
        for index in np.ndindex(value.shape):
            flat_values.append(_authority_value(value[index]))
        return (
            "ndarray",
            value.dtype.str,
            tuple(value.shape),
            tuple(flat_values),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(
                (field.name, _authority_value(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (
                        (_authority_value(key), _authority_value(item))
                        for key, item in value.items()
                    ),
                    key=repr,
                )
            ),
        )
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_authority_value(item) for item in value))
    raise TypeError(f"unsupported simulation authority value: {type(value).__name__}")


def simulation_authority_identity(
    plan: VariationPlan,
    config: SimulationConfig,
    compute_sensitivity: bool,
) -> object:
    """Return a lossless same-runtime identity for every execution input."""
    if not isinstance(plan, VariationPlan):
        raise TypeError("plan must be a VariationPlan")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if type(compute_sensitivity) is not bool:
        raise TypeError("compute_sensitivity must be bool")
    return _authority_value((plan, config, compute_sensitivity, plan.resolved_base()))


__all__ = [
    "AnnouncementRole",
    "VariationVisualEvent",
    "VariationVisualOrigin",
    "VariationVisualPhase",
    "VariationVisualState",
    "parse_variation_visual_state_matrix",
    "simulation_authority_identity",
    "variation_visual_state",
]
