"""Numeric primitives retained from the qualified rotating-base source."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

UPSTREAM_PHYSICS_SOURCE_SHA256 = (
    "a08641054a5ec58eaa9023ff123464c960b10833826f7ac9ba8dea68c26ab0d0"
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
N_COORDINATES = 7
N_CONSTRAINTS = 4


def _direction(angle: float) -> FloatArray:
    return np.array([np.sin(angle), -np.cos(angle)], dtype=float)


def _direction_derivative(angle: float) -> FloatArray:
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def _rotate(vector: npt.ArrayLike, angle: float) -> FloatArray:
    x, y = np.asarray(vector, dtype=float)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([cosine * x - sine * y, sine * x + cosine * y])


def _rotate_derivative(vector: npt.ArrayLike, angle: float) -> FloatArray:
    x, y = np.asarray(vector, dtype=float)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([-sine * x - cosine * y, cosine * x - sine * y])


def _finite_vector(name: str, value: object, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}")
    return array.copy()
