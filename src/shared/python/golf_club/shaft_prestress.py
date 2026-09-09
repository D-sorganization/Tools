"""Prescribed tensile beam operator: a partial reference for Tools #5072.

This module does not implement a general rotating shaft or impact solver.
The radial centrifugal load is a steady, straight-shaft equilibrium reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._beam_fem import assemble_bending_axis, generalized_eigenvalues
from ._validation import require_finite_float
from .shaft_dynamics import ShaftModalResponse, ShaftModalSettings
from .shaft_profile import ShaftProfile

_DEFAULT_SETTINGS = ShaftModalSettings()
# Four-point Gauss integrates N(s) B'(s)^T B'(s) exactly on each density segment:
# cubic N times quartic slope products has degree seven.
_GAUSS_POINTS, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)
_ASSUMPTIONS = (
    "small-deflection Euler-Bernoulli beam with prescribed tensile stress",
    "clamped exposed butt; optional point tip mass without rotary inertia",
    "midpoint EI and consistent distributed mass; no damping",
    "principal bending axes aligned with shaft frame; no spine rotation",
    "radial straight-shaft steady centrifugal load; axial strain not qualified",
    "no Coriolis, Euler, spin-softening, grip or head-mode operators",
    "rotating frequency interpretation restricted to out-of-plane radial limit",
    "inserted shaft mass is excluded; account for attached mass explicitly",
)


@dataclass(frozen=True)
class ShaftPrestress:
    """Explicit SI loading of an exposed shaft measured from its trimmed butt.

    Angular speed may have either sign; centrifugal tension is even in speed.
    Hub radius, point tip mass and additional dead tip tension are nonnegative.
    Compression, follower forces and time-dependent rotation are unsupported.
    """

    angular_speed_rad_s: float = 0.0
    hub_radius_m: float = 0.0
    tip_mass_kg: float = 0.0
    tip_tension_n: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "angular_speed_rad_s",
            "hub_radius_m",
            "tip_mass_kg",
            "tip_tension_n",
        ):
            value = require_finite_float(getattr(self, name), name)
            if name != "angular_speed_rad_s" and value < 0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)


def _validate(profile: ShaftProfile, prestress: ShaftPrestress) -> None:
    if not isinstance(profile, ShaftProfile):
        raise TypeError("profile must be ShaftProfile")
    if not isinstance(prestress, ShaftPrestress):
        raise TypeError("prestress must be ShaftPrestress")


def _boundaries(profile: ShaftProfile, start: float, end: float) -> np.ndarray:
    positions = [start, end]
    positions.extend(
        station.position_m - profile.butt_trim_m
        for station in profile.stations
        if start < station.position_m - profile.butt_trim_m < end
    )
    return np.sort(positions)


def radial_shaft_tension(
    profile: ShaftProfile, prestress: ShaftPrestress, position_m: float
) -> float:
    """Return tensile force N(s) from exposed outboard mass and tip load.

    Integrates piecewise-linear density times hub distance analytically.
    Position is measured along the exposed span, not the raw-shaft datum.
    The result is finite, nonnegative and includes no implicit inserted mass.
    """
    _validate(profile, prestress)
    position = require_finite_float(position_m, "position_m")
    length = profile.flexible_length_m
    if not 0 <= position <= length:
        raise ValueError("position_m must lie within the exposed span")
    points = _boundaries(profile, position, length)
    density = np.array(
        [
            profile.station_at(float(point) + profile.butt_trim_m).linear_density_kg_m
            for point in points
        ]
    )
    width = np.diff(points)
    # Product of two linear factors integrated exactly, without subtracting
    # nearby antiderivatives or dividing by a vanishing tip-segment length.
    radius = prestress.hub_radius_m + points
    outboard_moment = np.sum(
        width
        / 6
        * (
            2 * density[:-1] * radius[:-1]
            + density[:-1] * radius[1:]
            + density[1:] * radius[:-1]
            + 2 * density[1:] * radius[1:]
        )
    )
    outboard_moment += prestress.tip_mass_kg * (prestress.hub_radius_m + length)
    with np.errstate(over="ignore", invalid="ignore"):
        force = (
            prestress.tip_tension_n
            + np.square(prestress.angular_speed_rad_s) * outboard_moment
        )
    if not math.isfinite(force):
        raise ValueError("prestress produces nonfinite tension")
    return float(force)


def _slope_shapes(coordinate: np.ndarray, length: float) -> np.ndarray:
    fraction = coordinate / length
    return np.column_stack(
        (
            6 * (fraction**2 - fraction) / length,
            1 - 4 * fraction + 3 * fraction**2,
            6 * (fraction - fraction**2) / length,
            -2 * fraction + 3 * fraction**2,
        )
    )


def _element_geometric(
    profile: ShaftProfile, prestress: ShaftPrestress, bounds: tuple[float, float]
) -> np.ndarray:
    start, end = bounds
    points = _boundaries(profile, start, end)
    half_width = np.diff(points) / 2
    midpoint = (points[:-1] + points[1:]) / 2
    positions = (midpoint[:, None] + half_width[:, None] * _GAUSS_POINTS).ravel()
    weights = (half_width[:, None] * _GAUSS_WEIGHTS).ravel()
    forces = np.array(
        [radial_shaft_tension(profile, prestress, float(point)) for point in positions]
    )
    slopes = _slope_shapes(positions - start, end - start)
    return np.asarray(slopes.T @ ((weights * forces)[:, None] * slopes))


def shaft_geometric_stiffness(
    profile: ShaftProfile,
    prestress: ShaftPrestress,
    settings: ShaftModalSettings = _DEFAULT_SETTINGS,
) -> np.ndarray:
    """Return unconstrained symmetric tensile stiffness for one bending axis.

    Coordinates alternate displacement (m) and slope (rad). Thus matrix blocks
    have different SI dimensions. Energy is q.T K_g q / 2 in joules. The
    operator is positive semidefinite; rigid transverse translation is null.
    Each call returns an independent matrix, with no mutable cached state.
    """
    _validate(profile, prestress)
    if not isinstance(settings, ShaftModalSettings):
        raise TypeError("settings must be ShaftModalSettings")
    count = settings.element_count
    length = profile.flexible_length_m / count
    result = np.zeros((2 * (count + 1), 2 * (count + 1)))
    for element in range(count):
        indices = slice(2 * element, 2 * element + 4)
        result[indices, indices] += _element_geometric(
            profile, prestress, (element * length, (element + 1) * length)
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("prestress produces nonfinite geometric stiffness")
    return result


def _frequencies(
    profile: ShaftProfile,
    prestress: ShaftPrestress,
    settings: ShaftModalSettings,
    stiffness_name: str,
) -> tuple[float, ...]:
    stiffness, mass = assemble_bending_axis(
        profile, settings.element_count, stiffness_name
    )
    stiffness += shaft_geometric_stiffness(profile, prestress, settings)
    mass[-2, -2] += prestress.tip_mass_kg
    eigenvalues = generalized_eigenvalues(stiffness[2:, 2:], mass[2:, 2:])
    selected = eigenvalues[: settings.mode_count]
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0):
        raise ValueError("prestress modal solve is not finite positive definite")
    return tuple(float(math.sqrt(value) / (2 * math.pi)) for value in selected)


def solve_prestressed_shaft_modes(
    profile: ShaftProfile,
    prestress: ShaftPrestress,
    settings: ShaftModalSettings = _DEFAULT_SETTINGS,
) -> ShaftModalResponse:
    """Solve prescribed-tension bending modes, with explicit restricted scope.

    This returns two alternative uncoupled bending axes under the same tension.
    It does not predict both in-plane and out-of-plane rotating frequencies.
    A full rotating model must also account for velocity and softening terms.
    Nonzero station spine angles are refused: they require coupled bending.
    """
    _validate(profile, prestress)
    if not isinstance(settings, ShaftModalSettings):
        raise TypeError("settings must be ShaftModalSettings")
    if any(station.spine_angle_rad != 0 for station in profile.stations):
        raise ValueError("nonzero spine_angle_rad requires coupled bending operators")
    return ShaftModalResponse(
        frequencies_x_hz=_frequencies(profile, prestress, settings, "ei_about_y_n_m2"),
        frequencies_y_hz=_frequencies(profile, prestress, settings, "ei_about_x_n_m2"),
        flexible_length_m=profile.flexible_length_m,
        element_count=settings.element_count,
        model_name="prescribed_tension_bending_fem/1",
        assumptions=_ASSUMPTIONS,
    )


__all__ = [
    "ShaftPrestress",
    "radial_shaft_tension",
    "shaft_geometric_stiffness",
    "solve_prestressed_shaft_modes",
]
