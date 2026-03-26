"""Tests for shared reference-frame educational operations."""

from __future__ import annotations

import numpy as np

from rotation_converter.reference_frame_operations import (
    compute_homogeneous_transform,
    compute_so3_so3_maps,
    compute_twist_frame_conversion,
)


def test_twist_frame_conversion_identity() -> None:
    transform = np.eye(4).tolist()
    twist = [0.0, 0.0, 1.0, 0.5, 0.0, 0.0]
    result = compute_twist_frame_conversion(transform=transform, twist=twist)
    assert result.operation == "twist_frame_conversion"
    assert np.allclose(result.results["output_twist"], twist)
    assert "Ad_T" in result.explanation_markdown


def test_homogeneous_transform_inverse_roundtrip() -> None:
    rotation_matrix = np.eye(3).tolist()
    translation = [1.0, -2.0, 3.5]
    result = compute_homogeneous_transform(
        rotation_matrix=rotation_matrix, translation=translation
    )
    transform = np.asarray(result.results["homogeneous_transform"], dtype=float)
    inverse = np.asarray(result.results["inverse_transform"], dtype=float)
    assert result.operation == "homogeneous_transform"
    assert np.allclose(transform @ inverse, np.eye(4), atol=1e-10)
    assert "T^-1" in result.explanation_markdown


def test_so3_so3_maps_hat_vee_roundtrip() -> None:
    so3_vector = [0.1, -0.2, 0.3]
    result = compute_so3_so3_maps(so3_vector=so3_vector)
    assert result.operation == "so3_so3_maps"
    assert np.allclose(result.results["so3_vee_vector"], so3_vector, atol=1e-10)
    assert "exp" in result.explanation_markdown
