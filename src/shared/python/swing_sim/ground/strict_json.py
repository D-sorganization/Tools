"""Fail-closed JSON document parsing for ground contracts."""

from __future__ import annotations

import json
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_object(text: str) -> dict[str, Any]:
    """Parse an object while rejecting duplicate keys at every depth."""
    if not isinstance(text, str):
        raise TypeError("ground contract JSON must be text")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("ground contract JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("ground contract JSON must be an object")
    return value


__all__ = ["strict_json_object"]
