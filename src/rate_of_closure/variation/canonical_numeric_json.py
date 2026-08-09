"""Deterministic JSON encoding for cross-runtime numeric contract payloads."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

_FLOAT_QUANTUM = Decimal("0.00000000001")


def _canonical_float_token(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical JSON requires finite floats")
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    rounded = Decimal.from_float(value).quantize(_FLOAT_QUANTUM, rounding=ROUND_HALF_UP)
    if rounded.is_zero():
        return "0"
    return format(rounded, "f").rstrip("0").rstrip(".")


def canonical_numeric_json(value: Any) -> str:
    """Serialize JSON-compatible data with stable fixed-point float tokens."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _canonical_float_token(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        items = (
            f"{json.dumps(key, ensure_ascii=False)}:"
            f"{canonical_numeric_json(value[key])}"
            for key in sorted(value)
        )
        return "{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_numeric_json(item) for item in value) + "]"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


__all__ = ["canonical_numeric_json"]
