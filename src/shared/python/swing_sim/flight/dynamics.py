"""Callable aerodynamic dynamics used by the native flight integrator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._constants import MIN_SPEED_THRESHOLD_M_S, NUMERICAL_EPSILON, RPM_TO_RAD_S
from .aerodynamics import capped_lift_coefficient, spin_ratio_lift_coefficient
from .types import LaunchConditions

FloatArray: TypeAlias = NDArray[np.float64]


def _ballistic_derivative(velocity: FloatArray, gravity: float) -> FloatArray:
    return cast(
        FloatArray,
        np.array(
            [velocity[0], velocity[1], velocity[2], 0.0, 0.0, -gravity],
            dtype=np.float64,
        ),
    )


def _unit_spin_axis(launch: LaunchConditions) -> FloatArray:
    spin = launch.get_spin_vector()
    magnitude = math.hypot(spin[0], spin[1], spin[2])
    value = spin / magnitude if magnitude > 0.0 else spin
    return cast(FloatArray, np.asarray(value, dtype=np.float64))


@dataclass(frozen=True)
class WaterlooDynamics:
    """Quadratic drag and power-law lift state derivative."""

    launch: LaunchConditions
    parameters: tuple[float, ...]

    def __call__(self, time_s: float, state: FloatArray) -> FloatArray:
        if time_s is None:
            raise ValueError("time_s must be provided")
        cd0, cd1, cd2, cl0, cl1, cl2, cl_max = self.parameters
        velocity = state[3:]
        relative = velocity - self.launch.get_wind_vector(time_s, state[:3])
        speed = math.hypot(relative[0], relative[1], relative[2])
        if speed < MIN_SPEED_THRESHOLD_M_S:
            return _ballistic_derivative(velocity, self.launch.gravity)
        unit_velocity = relative / speed
        spin = self.launch.get_spin_vector()
        spin_magnitude = math.hypot(spin[0], spin[1], spin[2])
        spin_ratio = spin_magnitude * self.launch.ball_radius / speed
        drag_coefficient = cd0 + cd1 * spin_ratio + cd2 * spin_ratio**2
        lift_value = cl0 + cl1 * spin_ratio**cl2 if spin_ratio > 0.0 else cl0
        lift_coefficient = min(cl_max, capped_lift_coefficient(lift_value))
        area = math.pi * self.launch.ball_radius**2
        scale = 0.5 * self.launch.air_density * speed**2 * area
        acceleration = (
            -(scale * drag_coefficient / self.launch.ball_mass) * unit_velocity
        )
        if spin_magnitude > 0.0:
            direction = np.cross(spin / spin_magnitude, unit_velocity)
            direction_norm = math.hypot(direction[0], direction[1], direction[2])
            if direction_norm > NUMERICAL_EPSILON:
                acceleration += (scale * lift_coefficient / self.launch.ball_mass) * (
                    direction / direction_norm
                )
        acceleration[2] -= self.launch.gravity
        return cast(
            FloatArray,
            np.asarray(np.concatenate((velocity, acceleration)), dtype=np.float64),
        )


@dataclass(frozen=True)
class ConstantCoefficientDynamics:
    """Constant drag/lift state derivative with exponential spin decay."""

    launch: LaunchConditions
    drag_coefficient: float
    lift_coefficient: float
    spin_decay: float

    def __call__(self, time_s: float, state: FloatArray) -> FloatArray:
        if time_s is None:
            raise ValueError("time_s must be provided")
        velocity = state[3:]
        relative = velocity - self.launch.get_wind_vector(time_s, state[:3])
        speed = math.hypot(relative[0], relative[1], relative[2])
        if speed < MIN_SPEED_THRESHOLD_M_S:
            return _ballistic_derivative(velocity, self.launch.gravity)
        unit_velocity = relative / speed
        area = math.pi * self.launch.ball_radius**2
        scale = 0.5 * self.launch.air_density * area * speed**2
        acceleration = (
            -scale * self.drag_coefficient / self.launch.ball_mass
        ) * unit_velocity
        omega = (
            self.launch.spin_rate * RPM_TO_RAD_S * math.exp(-self.spin_decay * time_s)
        )
        if omega > 0.0:
            ratio = omega * self.launch.ball_radius / speed
            lift = spin_ratio_lift_coefficient(ratio, self.lift_coefficient)
            direction = np.cross(_unit_spin_axis(self.launch), unit_velocity)
            direction_norm = math.hypot(direction[0], direction[1], direction[2])
            if direction_norm > NUMERICAL_EPSILON:
                acceleration += (scale * lift / self.launch.ball_mass) * (
                    direction / direction_norm
                )
        acceleration[2] -= self.launch.gravity
        return cast(
            FloatArray,
            np.asarray(np.concatenate((velocity, acceleration)), dtype=np.float64),
        )


__all__ = ["ConstantCoefficientDynamics", "WaterlooDynamics"]
