"""Auditable modal finite-element reference for flexible golf shafts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .shaft_profile import ShaftProfile

_MODEL_NAME = "euler_bernoulli_bending_fem/1"
_ASSUMPTIONS = (
    "small transverse deflection",
    "Euler-Bernoulli bending without shear deformation or rotary inertia",
    "clamped trimmed butt and free exposed tip",
    "consistent distributed shaft mass without head or grip coupling",
    "undamped natural frequencies; stored damping is not applied",
)


@dataclass(frozen=True)
class ShaftModalSettings:
    """Validated spatial resolution and retained-mode count."""

    element_count: int = 16
    mode_count: int = 3

    def __post_init__(self) -> None:
        for name in ("element_count", "mode_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.element_count < 2:
            raise ValueError("element_count must be >= 2")
        if self.mode_count < 1:
            raise ValueError("mode_count must be >= 1")
        if self.mode_count > 2 * self.element_count:
            raise ValueError(
                "mode_count exceeds the available bending degrees of freedom"
            )


_DEFAULT_MODAL_SETTINGS = ShaftModalSettings()


@dataclass(frozen=True)
class ShaftModalResponse:
    """Undamped natural frequencies for both transverse shaft-frame axes.

    ``frequencies_x_hz`` describes deflection along shaft-frame x and
    therefore uses bending stiffness about y. ``frequencies_y_hz`` uses
    bending stiffness about x.
    """

    frequencies_x_hz: tuple[float, ...]
    frequencies_y_hz: tuple[float, ...]
    flexible_length_m: float
    element_count: int
    model_name: str = _MODEL_NAME
    assumptions: tuple[str, ...] = _ASSUMPTIONS

    def __post_init__(self) -> None:
        if self.element_count < 2:
            raise ValueError("element_count must be >= 2")
        if not math.isfinite(self.flexible_length_m) or self.flexible_length_m <= 0:
            raise ValueError("flexible_length_m must be finite and > 0")
        for name in ("frequencies_x_hz", "frequencies_y_hz"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a nonempty tuple")
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"{name} must contain finite positive values")
            if values != tuple(sorted(values)):
                raise ValueError(f"{name} must be sorted")


def solve_shaft_bending_modes(
    profile: ShaftProfile,
    settings: ShaftModalSettings = _DEFAULT_MODAL_SETTINGS,
) -> ShaftModalResponse:
    """Solve two uncoupled Euler-Bernoulli bending eigenproblems.

    Station properties are interpolated at element midpoints. A standard
    two-node cubic beam element and its consistent mass matrix are assembled
    in SI units. This is a verification-oriented linear reference, not a
    nonlinear swing transient or a calibrated composite-laminate model.
    """
    if not isinstance(profile, ShaftProfile):
        raise TypeError("profile must be ShaftProfile")
    if not isinstance(settings, ShaftModalSettings):
        raise TypeError("settings must be ShaftModalSettings")
    frequencies_x = _solve_axis(
        profile,
        settings,
        stiffness_name="ei_about_y_n_m2",
    )
    frequencies_y = _solve_axis(
        profile,
        settings,
        stiffness_name="ei_about_x_n_m2",
    )
    return ShaftModalResponse(
        frequencies_x_hz=frequencies_x,
        frequencies_y_hz=frequencies_y,
        flexible_length_m=profile.flexible_length_m,
        element_count=settings.element_count,
    )


def _solve_axis(
    profile: ShaftProfile,
    settings: ShaftModalSettings,
    *,
    stiffness_name: str,
) -> tuple[float, ...]:
    element_count = settings.element_count
    element_length = profile.flexible_length_m / element_count
    degrees = 2 * (element_count + 1)
    stiffness: np.ndarray = np.zeros((degrees, degrees), dtype=float)
    mass: np.ndarray = np.zeros((degrees, degrees), dtype=float)
    start = profile.butt_trim_m
    for element in range(element_count):
        midpoint = start + (element + 0.5) * element_length
        station = profile.station_at(midpoint)
        local_stiffness = _beam_stiffness(
            float(getattr(station, stiffness_name)), element_length
        )
        local_mass = _beam_mass(station.linear_density_kg_m, element_length)
        indices = np.array(
            [2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3]
        )
        stiffness[np.ix_(indices, indices)] += local_stiffness
        mass[np.ix_(indices, indices)] += local_mass
    eigenvalues = _generalized_eigenvalues(stiffness[2:, 2:], mass[2:, 2:])
    positive = eigenvalues[eigenvalues > np.finfo(float).eps]
    if len(positive) < settings.mode_count:
        raise RuntimeError("modal solve did not return enough positive eigenvalues")
    return tuple(
        float(math.sqrt(value) / (2.0 * math.pi))
        for value in positive[: settings.mode_count]
    )


def _beam_stiffness(ei_n_m2: float, length_m: float) -> np.ndarray:
    length_squared = length_m**2
    result: np.ndarray = np.asarray(
        ei_n_m2
        / length_m**3
        * np.array(
            [
                [12.0, 6.0 * length_m, -12.0, 6.0 * length_m],
                [
                    6.0 * length_m,
                    4.0 * length_squared,
                    -6.0 * length_m,
                    2.0 * length_squared,
                ],
                [-12.0, -6.0 * length_m, 12.0, -6.0 * length_m],
                [
                    6.0 * length_m,
                    2.0 * length_squared,
                    -6.0 * length_m,
                    4.0 * length_squared,
                ],
            ]
        )
    )
    return result


def _beam_mass(linear_density_kg_m: float, length_m: float) -> np.ndarray:
    length_squared = length_m**2
    result: np.ndarray = np.asarray(
        linear_density_kg_m
        * length_m
        / 420.0
        * np.array(
            [
                [156.0, 22.0 * length_m, 54.0, -13.0 * length_m],
                [
                    22.0 * length_m,
                    4.0 * length_squared,
                    13.0 * length_m,
                    -3.0 * length_squared,
                ],
                [54.0, 13.0 * length_m, 156.0, -22.0 * length_m],
                [
                    -13.0 * length_m,
                    -3.0 * length_squared,
                    -22.0 * length_m,
                    4.0 * length_squared,
                ],
            ]
        )
    )
    return result


def _generalized_eigenvalues(stiffness: np.ndarray, mass: np.ndarray) -> np.ndarray:
    factor = np.linalg.cholesky(mass)
    left_solved = np.linalg.solve(factor, stiffness)
    transformed = np.linalg.solve(factor, left_solved.T).T
    symmetric = 0.5 * (transformed + transformed.T)
    eigenvalues: np.ndarray = np.asarray(np.linalg.eigvalsh(symmetric))
    return eigenvalues


__all__ = [
    "ShaftModalResponse",
    "ShaftModalSettings",
    "solve_shaft_bending_modes",
]
