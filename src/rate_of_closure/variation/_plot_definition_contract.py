"""Strict scalar and identifier parsing for variation plot definitions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import cast

from shared.python.contracts import require


def _strict_integer(value: object, name: str) -> int:
    """Return one genuine JSON integer, excluding booleans."""
    require(type(value) is int, f"{name} must be an integer", value)
    return cast(int, value)


def _strict_nullable_string(value: object, name: str) -> str | None:
    """Return null or one non-empty trimmed string."""
    require(
        value is None or (isinstance(value, str) and _stable_string(value)),
        f"{name} must be null or a stable non-empty trimmed control-free string",
        value,
    )
    return cast(str | None, value)


def _strict_nullable_real(value: object, name: str) -> float | None:
    """Return null or one finite JSON real, excluding booleans."""
    if value is None:
        return None
    require(
        type(value) in {int, float},
        f"{name} must be null or a finite real number",
        value,
    )
    return _finite_float(cast(int | float, value), name)


def _strict_nullable_integer(value: object, name: str) -> int | None:
    """Return null or one genuine JSON integer."""
    require(value is None or type(value) is int, f"{name} must be null or integer")
    return cast(int | None, value)


def _normalize_nullable_real(value: object, name: str) -> float | None:
    """Normalize one finite constructor-domain real to JSON-native float."""
    require(
        value is None or (not isinstance(value, bool) and isinstance(value, Real)),
        f"{name} must be null or a finite real number",
        value,
    )
    return None if value is None else _finite_float(cast(Real, value), name)


def _normalize_nullable_integer(value: object, name: str) -> int | None:
    """Normalize one constructor-domain integral to JSON-native int."""
    require(
        value is None or (not isinstance(value, bool) and isinstance(value, Integral)),
        f"{name} must be null or integer",
        value,
    )
    return None if value is None else int(cast(Integral, value))


def _strict_variable_keys(value: object) -> tuple[str, ...] | None:
    """Return null or an exact string-array tuple."""
    require(value is None or isinstance(value, list), "variable_keys must be an array")
    if value is None:
        return None
    items = cast(list[object], value)
    require(
        _stable_strings(items),
        "variable_keys must contain trimmed control-free strings",
    )
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
            "variable_keys must contain non-empty trimmed control-free strings",
        )


def _validate_exact_fields(document: Mapping[str, object], expected: set[str]) -> None:
    """Reject omitted and unknown wire fields symmetrically."""
    require(set(document) == expected, "invalid plot definition fields")


def _stable_strings(items: list[object] | tuple[object, ...]) -> bool:
    """Return whether every item is one stable, non-empty identifier."""
    return all(isinstance(item, str) and _stable_string(item) for item in items)


def _stable_string(value: str) -> bool:
    """Reject whitespace instability and JSON-valid but unsafe control identifiers."""
    return (
        bool(value)
        and value == value.strip()
        and not any(ord(char) <= 0x1F or 0x7F <= ord(char) <= 0x9F for char in value)
    )


def _finite_float(value: object, name: str) -> float:
    """Convert a declared real without leaking conversion-domain exceptions."""
    try:
        result = float(cast(float, value))
    except (OverflowError, TypeError, ValueError):
        require(False, f"{name} must be representable as a finite JSON number", value)
        raise AssertionError("unreachable after contract violation") from None
    require(math.isfinite(result), f"{name} must be finite", value)
    return result


__all__ = [
    "_strict_integer",
    "_normalize_nullable_integer",
    "_normalize_nullable_real",
    "_strict_nullable_integer",
    "_strict_nullable_real",
    "_strict_nullable_string",
    "_strict_variable_keys",
    "_validate_exact_fields",
    "_validate_variable_keys_object",
]
