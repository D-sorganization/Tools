"""Built-in providers for the coordinate-explicit force-attribution core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .reference import coriolis_vector, damping_vector, gravity_vector, mass_matrix
from .types import PendulumParameters

FloatArray = NDArray[np.float64]


def _vector(name: str, value: object) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape (2,)")
    return result


@dataclass(frozen=True)
class DoublePendulumAttributionProvider:
    """Exact Tools double-pendulum provider in relative-angle coordinates."""

    parameters: PendulumParameters
    g_inplane: tuple[float, float]
    coordinate_names: tuple[str, ...] = ("shoulder_absolute", "wrist_relative")
    endpoint_name: str = "wrist_hand_path"

    def __post_init__(self) -> None:
        gravity = _vector("g_inplane", self.g_inplane)
        object.__setattr__(self, "g_inplane", (float(gravity[0]), float(gravity[1])))

    def mass_matrix(self, q: FloatArray) -> FloatArray:
        q_array = _vector("q", q)
        return np.asarray(mass_matrix(self.parameters, float(q_array[1])))

    def mass_matrix_derivatives(self, q: FloatArray) -> FloatArray:
        q_array = _vector("q", q)
        coupling = self.parameters.m2 * self.parameters.l1 * self.parameters.lc2
        derivative = -coupling * np.sin(q_array[1])
        result = np.zeros((2, 2, 2), dtype=np.float64)
        result[1] = np.array([[2.0 * derivative, derivative], [derivative, 0.0]])
        return result

    def velocity_bias(self, q: FloatArray, velocity: FloatArray) -> FloatArray:
        q_array = _vector("q", q)
        speed = _vector("velocity", velocity)
        return np.asarray(
            coriolis_vector(
                self.parameters,
                float(q_array[1]),
                float(speed[0]),
                float(speed[1]),
            ),
            dtype=np.float64,
        )

    def gravity(self, q: FloatArray) -> FloatArray:
        q_array = _vector("q", q)
        return np.asarray(
            gravity_vector(
                self.parameters,
                float(q_array[0]),
                float(q_array[1]),
                self.g_inplane,
            ),
            dtype=np.float64,
        )

    def damping(self, velocity: FloatArray) -> FloatArray:
        speed = _vector("velocity", velocity)
        return np.asarray(
            damping_vector(self.parameters, float(speed[0]), float(speed[1])),
            dtype=np.float64,
        )

    def endpoint_jacobian(self, q: FloatArray) -> FloatArray:
        q_array = _vector("q", q)
        theta = float(q_array[0])
        return np.array(
            [
                [self.parameters.l1 * np.cos(theta), 0.0],
                [self.parameters.l1 * np.sin(theta), 0.0],
            ],
            dtype=np.float64,
        )


__all__ = ["DoublePendulumAttributionProvider"]
