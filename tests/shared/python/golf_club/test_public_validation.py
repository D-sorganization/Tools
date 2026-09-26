"""Tests for public rotation and inertia validators (Tools #5353 item 4)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.golf_club._validation import (
    require_inertia as _private_require_inertia,
)
from shared.python.golf_club._validation import (
    require_rotation as _private_require_rotation,
)
from shared.python.golf_club.validation import (
    require_inertia,
    require_rotation,
)

pytestmark = [pytest.mark.unit]


def test_public_validators_are_same_objects_as_private() -> None:
    """Public validators must re-export private validators without wrapping."""
    assert require_rotation is _private_require_rotation
    assert require_inertia is _private_require_inertia


def test_require_rotation_accepts_identity() -> None:
    """Identity matrix must pass rotation validation."""
    identity = np.eye(3)
    result = require_rotation(identity)
    np.testing.assert_allclose(result, identity)


def test_require_rotation_rejects_reflection() -> None:
    """Reflection with determinant -1 must be rejected."""
    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="proper orthonormal"):
        require_rotation(reflection)


def test_require_rotation_rejects_non_orthonormal_matrix() -> None:
    """Non-orthonormal matrix must be rejected."""
    non_orthonormal = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="proper orthonormal"):
        require_rotation(non_orthonormal)


def test_require_inertia_accepts_valid_diagonal_tensor() -> None:
    """Valid diagonal inertia tensor satisfying triangle inequality must pass."""
    tensor = ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.5))
    result = require_inertia(tensor)
    assert result == tensor


def test_require_inertia_rejects_asymmetric_tensor() -> None:
    """Asymmetric tensor must be rejected with ValueError."""
    asymmetric = ((1.0, 0.1, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.5))
    with pytest.raises(ValueError, match="must be symmetric"):
        require_inertia(asymmetric)


def test_require_inertia_rejects_negative_eigenvalue() -> None:
    """Tensor with negative eigenvalue must be rejected as not PSD."""
    not_psd = ((-0.1, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.5))
    with pytest.raises(ValueError, match="positive semidefinite"):
        require_inertia(not_psd)


def test_require_inertia_rejects_triangle_inequality_violation() -> None:
    """Tensor violating principal moment triangle inequality must be rejected."""
    # 4.0 > 1.0 + 2.0 = 3.0
    violates_triangle = ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 4.0))
    with pytest.raises(ValueError, match="triangle inequality"):
        require_inertia(violates_triangle)
