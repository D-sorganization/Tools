"""Strict control-free identity contracts for variation models and JSON wires."""

from __future__ import annotations

from typing import cast

from shared.python.contracts import require


def strict_string(value: object, name: str) -> str:
    """Require a real string without coercing another JSON primitive."""
    require(isinstance(value, str), f"{name} must be a string", value)
    return cast(str, value)


def stable_id(value: object, name: str) -> str:
    """Require a nonempty, trimmed, C0/C1-control-free stable identifier."""
    text = strict_string(value, name)
    require(
        bool(text)
        and text == text.strip()
        and all(
            ord(character) >= 32 and not 127 <= ord(character) <= 159
            for character in text
        ),
        f"{name} must be a non-empty, trimmed, control-free stable ID",
        value,
    )
    return text


def stable_id_array(value: object, name: str) -> tuple[str, ...]:
    """Parse one JSON array of unique stable IDs without scalar iteration."""
    require(type(value) is list, f"{name} must be an array", value)
    result = tuple(stable_id(item, name) for item in cast(list[object], value))
    require(len(set(result)) == len(result), f"{name} must be unique", result)
    return result


__all__ = ["stable_id", "stable_id_array", "strict_string"]
