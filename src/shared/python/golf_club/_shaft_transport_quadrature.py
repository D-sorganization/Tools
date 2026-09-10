"""Section motion mapping and property quadrature for a straight Rayleigh rod."""

from __future__ import annotations

import numpy as np

from ._shaft_transport_contracts import ShaftRotatingModel
from ._validation import require_inertia
from .shaft_prestress import _boundaries, _slope_shapes
from .shaft_profile import ShaftProfile
from .types import ComponentMassProperties, ComponentRole

_GAUSS_POINTS, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)


def element_quadrature(
    profile: ShaftProfile, bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    points = _boundaries(profile, *bounds)
    half_width = np.diff(points) / 2
    midpoint = (points[:-1] + points[1:]) / 2
    positions = (midpoint[:, None] + half_width[:, None] * _GAUSS_POINTS).ravel()
    weights = (half_width[:, None] * _GAUSS_WEIGHTS).ravel()
    return positions, weights


def section_motion_map(coordinate: float, length: float) -> np.ndarray:
    """Map twelve nodal DOFs to section [u, theta] with theta_y=ux', theta_x=-uy'."""
    fraction = coordinate / length
    bending = np.array(
        [
            1 - 3 * fraction**2 + 2 * fraction**3,
            length * (fraction - 2 * fraction**2 + fraction**3),
            3 * fraction**2 - 2 * fraction**3,
            length * (-(fraction**2) + fraction**3),
        ]
    )
    slope = _slope_shapes(np.array([coordinate]), length)[0]
    x_indices, y_indices = [0, 4, 6, 10], [1, 3, 7, 9]
    signs = np.array([1, -1, 1, -1])
    result = np.zeros((6, 12))
    result[0, x_indices], result[1, y_indices] = bending, bending * signs
    result[3, y_indices], result[4, x_indices] = -slope * signs, slope
    result[2, [2, 8]], result[5, [5, 11]] = (
        (1 - fraction, fraction),
        (1 - fraction, fraction),
    )
    return result


def quadrature_body(
    model: ShaftRotatingModel, position: float, weight: float
) -> ComponentMassProperties:
    """Convert density times quadrature length to a finite physical body sample."""
    profile = model.rod.profile
    raw = position + profile.butt_trim_m
    station = profile.station_at(raw)
    positions = [station.position_m for station in profile.stations]
    transverse = np.asarray(model.rotary_inertia.transverse_kg_m)
    xx, yy, xy = [float(np.interp(raw, positions, column)) for column in transverse.T]
    inertia = weight * np.array([[xx, xy, 0], [xy, yy, 0], [0, 0, xx + yy]])
    return ComponentMassProperties(
        "quadrature-section",
        ComponentRole.SHAFT,
        profile.frame_id,
        station.linear_density_kg_m * weight,
        (0, 0, 0),
        require_inertia(inertia),
    )
