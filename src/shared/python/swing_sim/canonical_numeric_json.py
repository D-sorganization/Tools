"""Deterministic JSON encoding for cross-runtime numeric contract payloads."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

_FLOAT_QUANTUM = Decimal("0.00000000001")
MAX_CANONICAL_SAFE_INTEGER = 9_007_199_254_740_991


def _string_token(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("canonical JSON does not permit surrogate code points")
    return json.dumps(value, ensure_ascii=False)


def _canonical_float_token(value: float, *, allow_extended_finite: bool = False) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical JSON requires finite floats")
    if not allow_extended_finite and abs(value) > MAX_CANONICAL_SAFE_INTEGER:
        raise ValueError("canonical JSON number exceeds cross-runtime safe range")
    if value == 0 or value.is_integer():
        return "0" if value == 0 else str(int(value))
    rounded = Decimal.from_float(value).quantize(_FLOAT_QUANTUM, rounding=ROUND_HALF_UP)
    if rounded.is_zero():
        return "0"
    return format(rounded, "f").rstrip("0").rstrip(".")


def canonical_numeric_float(value: float) -> float:
    """Return the finite float represented by the canonical 11-place token."""
    return float(_canonical_float_token(value))


def _canonical_numeric_json(value: Any, *, allow_extended_finite: bool) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _canonical_float_token(
            value, allow_extended_finite=allow_extended_finite
        )
    if isinstance(value, int):
        if abs(value) > MAX_CANONICAL_SAFE_INTEGER:
            raise ValueError("canonical JSON integer exceeds cross-runtime safe range")
        return str(value)
    if isinstance(value, str):
        return _string_token(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")

        def item_token(key: str) -> str:
            token = _canonical_numeric_json(
                value[key], allow_extended_finite=allow_extended_finite
            )
            return f"{_string_token(key)}:{token}"

        items = (item_token(key) for key in sorted(value))
        return "{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        items = (
            _canonical_numeric_json(item, allow_extended_finite=allow_extended_finite)
            for item in value
        )
        return "[" + ",".join(items) + "]"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_numeric_json(value: Any) -> str:
    """Serialize JSON-compatible data within the shared runtime-safe range."""
    return _canonical_numeric_json(value, allow_extended_finite=False)


def canonical_numeric_json_extended_floats(value: Any) -> str:
    """Serialize finite floats beyond the safe range while retaining safe integers."""
    return _canonical_numeric_json(value, allow_extended_finite=True)


__all__ = [
    "MAX_CANONICAL_SAFE_INTEGER",
    "canonical_numeric_float",
    "canonical_numeric_json",
    "canonical_numeric_json_extended_floats",
]
