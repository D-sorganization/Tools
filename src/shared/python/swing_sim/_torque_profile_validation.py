"""Private validation helpers for the prescribed torque profile schema."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Any, cast

from shared.python.contracts import require

_STABLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_METADATA_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)


def strict_mapping(
    data: object, fields: frozenset[str], label: str
) -> Mapping[str, Any]:
    """Validate an exact JSON object shape and return it as a mapping."""
    require(isinstance(data, Mapping), f"{label} must be a JSON object", data)
    mapping = cast(Mapping[str, Any], data)
    actual = frozenset(mapping.keys())
    require(actual == fields, f"{label} fields must match the schema exactly", actual)
    return mapping


def finite_float(value: object, label: str) -> float:
    """Convert one real numeric value and require finiteness."""
    require(not isinstance(value, bool), f"{label} must be a real number", value)
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        require(False, f"{label} must be a real number", value)
        raise AssertionError("unreachable") from None
    require(math.isfinite(converted), f"{label} must be finite", value)
    return converted


def finite_tuple(values: object, label: str) -> tuple[float, ...]:
    """Normalize a non-string sequence of finite real values."""
    require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes)),
        f"{label} must be a sequence",
        values,
    )
    sequence = cast(Sequence[object], values)
    normalized = tuple(finite_float(value, label) for value in sequence)
    require(len(normalized) > 0, f"{label} must not be empty", values)
    return normalized


def stable_id(value: object, label: str) -> str:
    """Validate a portable, display-label-independent identifier."""
    require(isinstance(value, str), f"{label} must be a string", value)
    identifier = cast(str, value)
    require(bool(_STABLE_ID_PATTERN.fullmatch(identifier)), f"invalid {label}", value)
    return identifier


def time_domain(values: object) -> tuple[float, float]:
    """Normalize a finite, strictly ordered two-value time domain."""
    domain = finite_tuple(values, "time_domain_s")
    require(len(domain) == 2, "time_domain_s must contain exactly two values")
    start_s, end_s = domain[0], domain[1]
    require(start_s < end_s, "time_domain_s must be strictly ordered", domain)
    return start_s, end_s


def source_metadata(value: object) -> Mapping[str, str]:
    """Freeze a string-only provenance mapping with portable field names."""
    require(isinstance(value, Mapping), "source_metadata must be a mapping", value)
    mapping = cast(Mapping[object, object], value)
    normalized: dict[str, str] = {}
    for key, item in mapping.items():
        require(
            isinstance(key, str) and bool(_METADATA_KEY_PATTERN.fullmatch(key)),
            "invalid source_metadata key",
            key,
        )
        require(
            isinstance(item, str) and bool(item.strip()),
            "source_metadata values must be nonempty strings",
            item,
        )
        normalized[cast(str, key)] = cast(str, item)
    require(len(normalized) > 0, "source_metadata must not be empty")
    return MappingProxyType(normalized)


def utc_timestamp_pair(created: object, modified: object) -> tuple[str, str]:
    """Validate canonical UTC timestamps and their chronological order."""
    created_text = _utc_timestamp(created, "created_at_utc")
    modified_text = _utc_timestamp(modified, "modified_at_utc")
    created_dt = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
    modified_dt = datetime.fromisoformat(modified_text.replace("Z", "+00:00"))
    require(modified_dt >= created_dt, "modified_at_utc must not precede creation")
    return created_text, modified_text


def _utc_timestamp(value: object, label: str) -> str:
    """Validate one canonical ISO-8601 UTC timestamp."""
    require(isinstance(value, str), f"{label} must be a string", value)
    text = cast(str, value)
    require(bool(_UTC_TIMESTAMP_PATTERN.fullmatch(text)), f"invalid {label}", value)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        require(False, f"invalid {label}", value)
    return text


def sha256_or_none(value: object) -> str | None:
    """Validate an optional lowercase SHA-256 digest."""
    require(
        value is None
        or (isinstance(value, str) and bool(_SHA256_PATTERN.fullmatch(value))),
        "original_sample_sha256 must be 64 lowercase hexadecimal characters",
        value,
    )
    return cast(str | None, value)


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting duplicate field names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field", key)
        result[key] = value
    return result


__all__ = [
    "finite_float",
    "finite_tuple",
    "sha256_or_none",
    "source_metadata",
    "stable_id",
    "strict_mapping",
    "time_domain",
    "unique_json_object",
    "utc_timestamp_pair",
]
