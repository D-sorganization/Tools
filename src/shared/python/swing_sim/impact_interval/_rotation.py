"""Small SO(3) helpers kept private behind the impact-interval façade."""

from __future__ import annotations

import math

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    """Cross-product matrix for a three-vector."""
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_rotation(rotation_vector: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map with a stable small-angle branch."""
    angle = float(np.linalg.norm(rotation_vector))
    matrix = skew(rotation_vector)
    if angle < 1.0e-10:
        return np.eye(3) + matrix + 0.5 * matrix @ matrix
    axis_matrix = matrix / angle
    return (
        np.eye(3)
        + math.sin(angle) * axis_matrix
        + (1.0 - math.cos(angle)) * axis_matrix @ axis_matrix
    )


def log_rotation(rotation: np.ndarray) -> np.ndarray:
    """Principal rotation vector for a proper 3x3 rotation matrix."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1.0e-9:
        return 0.5 * vector
    return angle * vector / (2.0 * math.sin(angle))


__all__: list[str] = []
