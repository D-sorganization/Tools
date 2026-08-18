"""Fail-closed v1-only migration gateways for ground profile documents."""

from __future__ import annotations

from typing import Any, cast

from .profile_types import GroundMaterialProfile, GroundProfileLibrary


def migrate_profile_to_current(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a v1 profile with no implicit predecessor."""
    profile = cast(GroundMaterialProfile, GroundMaterialProfile.from_dict(payload))
    return profile.to_dict()


def migrate_library_to_current(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a v1 library with no implicit predecessor."""
    library = cast(GroundProfileLibrary, GroundProfileLibrary.from_dict(payload))
    return library.to_dict()


__all__ = ["migrate_library_to_current", "migrate_profile_to_current"]
