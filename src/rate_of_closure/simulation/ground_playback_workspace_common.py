"""Shared structural validation for versioned ground playback workspaces."""

from __future__ import annotations

from typing import Any, cast


def exact_workspace_fields(
    payload: dict[str, Any],
    expected: set[str],
    name: str,
    schema_label: str,
) -> None:
    """Require an exact object field set for one named workspace schema."""
    if set(payload) != expected:
        raise ValueError(f"{name} fields do not match {schema_label} schema")


def workspace_object(value: object, name: str) -> dict[str, Any]:
    """Return a string-keyed JSON object or reject the boundary value."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


__all__ = ["exact_workspace_fields", "workspace_object"]
