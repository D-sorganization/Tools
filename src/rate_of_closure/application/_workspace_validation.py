"""Strict JSON and identity primitives for workspace persistence."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Any, TypeAlias

from shared.python.compatibility import UTC

_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

JsonScalar: TypeAlias = str | int | float | bool | None
FrozenJsonValue: TypeAlias = (
    JsonScalar | tuple["FrozenJsonValue", ...] | Mapping[str, "FrozenJsonValue"]
)


def exact_mapping(
    value: object, expected_fields: frozenset[str], name: str
) -> Mapping[str, Any]:
    """Return a mapping only when its string keys exactly match the contract."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    actual = frozenset(value)
    if actual != expected_fields:
        missing = sorted(expected_fields - actual)
        unknown = sorted(actual - expected_fields)
        raise ValueError(
            f"{name} fields mismatch; missing={missing}, unknown={unknown}"
        )
    return value


def stable_id(value: object, name: str) -> str:
    """Return a bounded portable identifier or fail closed."""
    if not isinstance(value, str) or _STABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a stable identifier")
    return value


def valid_stable_id(value: object) -> bool:
    """Return whether a value is a portable stable identifier."""
    return isinstance(value, str) and _STABLE_ID.fullmatch(value) is not None


def positive_version(value: object, name: str) -> int:
    """Return a non-boolean positive integer schema version."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def utc_datetime(value: object, name: str) -> datetime:
    """Parse a strict ISO-8601 UTC timestamp ending in ``Z``."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo != UTC:
        raise ValueError(f"{name} must use UTC")
    return parsed


def _freeze_json(value: object, path: str) -> FrozenJsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, list):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} field names must be strings")
        frozen = {
            key: _freeze_json(item, f"{path}.{key}") for key, item in value.items()
        }
        return MappingProxyType(frozen)
    raise TypeError(f"{path} contains unsupported JSON value {type(value).__name__}")


def freeze_object(value: object, name: str) -> Mapping[str, FrozenJsonValue]:
    """Validate and recursively freeze one strict JSON object."""
    frozen = _freeze_json(value, name)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return frozen


def thaw_json(value: FrozenJsonValue) -> Any:
    """Return a detached mutable JSON-compatible value."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


__all__ = [
    "FrozenJsonValue",
    "exact_mapping",
    "freeze_object",
    "positive_version",
    "stable_id",
    "thaw_json",
    "unique_json_object",
    "utc_datetime",
    "valid_stable_id",
]
