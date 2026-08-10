"""Deterministic JSON encoding for cross-runtime numeric contract payloads."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

_FLOAT_QUANTUM = Decimal("0.00000000001")
_MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _string_token(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("canonical JSON does not permit surrogate code points")
    return json.dumps(value, ensure_ascii=False)


def _canonical_float_token(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical JSON requires finite floats")
    if value == 0 or value.is_integer():
        return "0" if value == 0 else str(int(value))
    rounded = Decimal.from_float(value).quantize(_FLOAT_QUANTUM, rounding=ROUND_HALF_UP)
    if rounded.is_zero():
        return "0"
    return format(rounded, "f").rstrip("0").rstrip(".")


def canonical_numeric_float(value: float) -> float:
    """Return the finite float represented by the canonical 11-place token."""
    return float(_canonical_float_token(value))


def canonical_numeric_json(value: Any) -> str:
    """Serialize JSON-compatible data with stable fixed-point float tokens."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _canonical_float_token(value)
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValueError("canonical JSON integer exceeds cross-runtime safe range")
        return str(value)
    if isinstance(value, str):
        return _string_token(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        items = (
            f"{_string_token(key)}:{canonical_numeric_json(value[key])}"
            for key in sorted(value)
        )
        return "{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_numeric_json(item) for item in value) + "]"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


__all__ = ["canonical_numeric_float", "canonical_numeric_json"]
