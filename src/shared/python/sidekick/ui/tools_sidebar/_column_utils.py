"""Shared column-resolution helper for sidebar data tabs."""

from __future__ import annotations

from collections.abc import Callable


def _resolve_columns(
    available: list[str],
    selected: list[str] | None,
    make_error: Callable[[list[str]], Exception],
) -> list[str]:
    """Return the effective column list, raising a domain error on unknown names.

    Preconditions: available is a list of str; make_error is callable.
    """
    if not isinstance(available, list):
        raise TypeError("available must be a list")
    if make_error is None or not callable(make_error):
        raise TypeError("make_error must be callable")
    if not selected:
        return available
    normalized = [col.strip() for col in selected if col.strip()]
    missing = [col for col in normalized if col not in available]
    if missing:
        raise make_error(missing)
    return normalized
