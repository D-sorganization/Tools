"""Primitive validators for the Morris observation archive."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, cast

import numpy as np

from shared.python.contracts import require


def exact_mapping(
    value: object, fields: frozenset[str], name: str
) -> Mapping[str, Any]:
    """Require an object with exactly the declared field set."""
    require(isinstance(value, Mapping), f"{name} must be an object", value)
    item = cast(Mapping[str, Any], value)
    require(
        all(isinstance(key, str) for key in item) and set(item) == fields,
        f"{name} must contain the exact field set",
        tuple(item),
    )
    return item


def stable_text(value: object, name: str, *, maximum: int = 1_024) -> str:
    """Require bounded, trimmed, control-free text."""
    require(
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and len(value) <= maximum
        and all(
            ord(character) >= 32 and not 127 <= ord(character) <= 159
            for character in value
        ),
        f"{name} must be a bounded nonempty trimmed string",
        value,
    )
    return cast(str, value)


def optional_text(value: object, name: str) -> str | None:
    """Validate optional stable text."""
    return None if value is None else stable_text(value, name)


def nonnegative_integer(value: object, name: str) -> int:
    """Require a true nonnegative integer, excluding booleans."""
    result = int(value) if type(value) is int else -1
    require(
        type(value) is int and result >= 0,
        f"{name} must be a nonnegative integer",
        value,
    )
    return result


def finite(value: object, name: str) -> float:
    """Require a finite real scalar, excluding booleans."""
    require(
        not isinstance(value, (bool, np.bool_)) and isinstance(value, Real),
        f"{name} must be finite",
        value,
    )
    result = float(cast(Real, value))
    require(math.isfinite(result), f"{name} must be finite", result)
    return result


def nullable_finite(value: object, name: str) -> float | None:
    """Validate an optional finite scalar."""
    return None if value is None else finite(value, name)


def sha256_hex(value: object, name: str) -> str:
    """Require lowercase SHA-256 hexadecimal text."""
    text = stable_text(value, name, maximum=64)
    require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{name} must be lowercase SHA-256 hexadecimal",
        text,
    )
    return text


def provenance_mapping(value: object) -> dict[str, str]:
    """Normalize sorted bounded string provenance."""
    require(
        isinstance(value, Mapping) and bool(value),
        "provenance must be a nonempty object",
        value,
    )
    result: dict[str, str] = {}
    for key, item in cast(Mapping[object, object], value).items():
        stable_key = stable_text(key, "provenance key", maximum=128)
        require(stable_key not in result, "provenance keys must be unique", stable_key)
        result[stable_key] = stable_text(item, f"provenance.{stable_key}")
    return dict(sorted(result.items()))


__all__ = [
    "exact_mapping",
    "finite",
    "nonnegative_integer",
    "nullable_finite",
    "optional_text",
    "provenance_mapping",
    "sha256_hex",
    "stable_text",
]
