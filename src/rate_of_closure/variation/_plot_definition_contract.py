"""Strict scalar and identifier parsing for variation plot definitions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import cast

from shared.python.contracts import require


def _strict_integer(value: object, name: str) -> int:
    """Return one genuine JSON integer, excluding booleans."""
    require(type(value) is int, f"{name} must be an integer", value)
    return cast(int, value)


def _strict_nullable_string(value: object, name: str) -> str | None:
    """Return null or one non-empty trimmed string."""
    require(
        value is None
        or (isinstance(value, str) and bool(value) and value == value.strip()),
        f"{name} must be null or a non-empty trimmed string",
        value,
    )
    return cast(str | None, value)


def _strict_nullable_real(value: object, name: str) -> float | None:
    """Return null or one finite JSON real, excluding booleans."""
    require(
        value is None
        or (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(float(value))
        ),
        f"{name} must be null or a finite real number",
        value,
    )
    return None if value is None else float(cast(float, value))


def _strict_nullable_integer(value: object, name: str) -> int | None:
    """Return null or one genuine JSON integer."""
    require(value is None or type(value) is int, f"{name} must be null or integer")
    return cast(int | None, value)


def _strict_variable_keys(value: object) -> tuple[str, ...] | None:
    """Return null or an exact string-array tuple."""
    require(value is None or isinstance(value, list), "variable_keys must be an array")
    if value is None:
        return None
    items = cast(list[object], value)
    require(_stable_strings(items), "variable_keys must contain trimmed strings")
    return tuple(cast(list[str], items))


def _validate_variable_keys_object(value: object) -> None:
    """Validate the immutable constructor-domain form before serialization."""
    require(
        value is None or isinstance(value, tuple),
        "variable_keys must be null or a tuple",
    )
    if value is not None:
        require(
            _stable_strings(cast(tuple[object, ...], value)),
            "variable_keys must contain non-empty trimmed strings",
        )


def _validate_exact_fields(document: Mapping[str, object], expected: set[str]) -> None:
    """Reject omitted and unknown wire fields symmetrically."""
    require(set(document) == expected, "invalid plot definition fields")


def _stable_strings(items: list[object] | tuple[object, ...]) -> bool:
    """Return whether every item is one stable, non-empty identifier."""
    return all(
        isinstance(item, str) and bool(item) and item == item.strip() for item in items
    )


__all__ = [
    "_strict_integer",
    "_strict_nullable_integer",
    "_strict_nullable_real",
    "_strict_nullable_string",
    "_strict_variable_keys",
    "_validate_exact_fields",
    "_validate_variable_keys_object",
]
