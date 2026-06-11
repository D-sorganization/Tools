"""Tests for shared rotation transform primitives."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
import numpy as np
from rotation_transforms import (
    Rotation,
    compute_homogeneous_transform,
    compute_reference_frame_operation,
    compute_twist_frame_conversion,
)


def test_rotation_quaternion_round_trips_through_matrix() -> None:
    rotation = Rotation.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)

    round_tripped = Rotation.from_rotation_matrix(rotation.as_rotation_matrix())

    assert np.allclose(round_tripped.as_quaternion(), rotation.as_quaternion())


def test_reference_frame_dispatch_builds_inverse_transform() -> None:
    result = compute_reference_frame_operation(
        "homogeneous_transform",
        rotation_matrix=np.eye(3),
        translation=[1.0, -2.0, 3.0],
    )

    transform = np.asarray(result.results["homogeneous_transform"], dtype=float)
    inverse = np.asarray(result.results["inverse_transform"], dtype=float)

    assert result.operation == "homogeneous_transform"
    assert np.allclose(transform @ inverse, np.eye(4))


def test_twist_frame_conversion_applies_translation_adjoint() -> None:
    transform_result = compute_homogeneous_transform(
        rotation_matrix=np.eye(3),
        translation=[0.0, 1.0, 0.0],
    )
    twist_result = compute_twist_frame_conversion(
        transform=transform_result.results["homogeneous_transform"],
        twist=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )

    assert np.allclose(
        twist_result.results["output_twist"], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    )


def test_rotation_is_hashable() -> None:
    """Test that Rotation instances are hashable and can be used in sets/dicts."""
    r1 = Rotation.identity()
    r2 = Rotation.identity()
    r3 = Rotation.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)

    # Equal rotations should have equal hashes
    assert hash(r1) == hash(r2)
    # Can add to sets
    rotation_set = {r1, r2, r3}
    assert len(rotation_set) == 2  # r1 and r2 are equal, so only 2 unique
    # Can use as dict keys
    rotation_dict = {r1: "identity", r3: "90deg_z"}
    assert rotation_dict[r2] == "identity"  # r2 equals r1
