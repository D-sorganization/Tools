"""Helpers for trimesh boolean heavy-integration contracts (installed-package lane)."""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "EXPECTED_TRANSLATED_BOX_INTERSECTION_VOLUME",
    "require_trimesh_boolean_backend",
]


# Two 1×1×1 boxes overlapping by 0.5 m along X → intersection volume 0.5 m³.
EXPECTED_TRANSLATED_BOX_INTERSECTION_VOLUME = 0.5


def require_trimesh_boolean_backend(
    trimesh_module: Any,
    *,
    skip: Callable[[str], Any],
) -> None:
    """Skip only when no boolean engine is provisioned; do not call intersection first."""
    if not trimesh_module.boolean.engines_available:
        skip(
            "Trimesh boolean backend not available "
            "(install manifold3d or provision blender)"
        )
