"""Supported public strict validators for rotation matrices and inertia tensors.

Tolerances enforced by these validators:
- ``_ROTATION_TOLERANCE``: 1e-10 (matrix orthonormality and proper determinant)
- ``_PSD_TOLERANCE``: 1e-12 (positive semidefiniteness and principal-moment
  triangle inequality)
- ``_SYMMETRY_TOLERANCE``: 1e-12 (tensor symmetry)
"""

from __future__ import annotations

from ._validation import (
    require_inertia,
    require_rotation,
)

__all__ = [
    "require_inertia",
    "require_rotation",
]
