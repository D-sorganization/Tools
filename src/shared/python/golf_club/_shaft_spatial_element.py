"""Unloaded six-axis straight beam kernel, sharing the existing bending FEM."""

from __future__ import annotations

import numpy as np

from ._beam_fem import beam_mass, beam_stiffness
from ._shaft_linear_contracts import ShaftRodProperties
from .shaft_profile import ShaftStation
from .types import ComponentMassProperties

_ROD_GRADIENT = np.array([[1, -1], [-1, 1]])
_ROD_MASS = np.array([[2, 1], [1, 2]]) / 6
_X_BENDING = np.array([0, 4, 6, 10])
_Y_BENDING = np.array([1, 3, 7, 9])
_Y_SIGNS = np.array([1, -1, 1, -1])


def _bending_matrices(
    station: ShaftStation, length: float
) -> tuple[np.ndarray, np.ndarray]:
    stiffness, mass = np.zeros((12, 12)), np.zeros((12, 12))
    for indices, ei, signs in (
        (_X_BENDING, station.ei_about_y_n_m2, np.ones(4)),
        (_Y_BENDING, station.ei_about_x_n_m2, _Y_SIGNS),
    ):
        sign_map = np.outer(signs, signs)
        stiffness[np.ix_(indices, indices)] = beam_stiffness(ei, length) * sign_map
        mass[np.ix_(indices, indices)] = (
            beam_mass(station.linear_density_kg_m, length) * sign_map
        )
    return stiffness, mass


def _principal_rotation(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    return np.kron(np.eye(4), rotation)


def spatial_element(
    rod: ShaftRodProperties, raw_midpoint: float, length: float
) -> tuple[np.ndarray, np.ndarray]:
    station = rod.profile.station_at(raw_midpoint)
    stiffness, mass = _bending_matrices(station, length)
    positions = [station.position_m for station in rod.profile.stations]
    axial = float(np.interp(raw_midpoint, positions, rod.axial_stiffness_n))
    polar = float(np.interp(raw_midpoint, positions, rod.polar_mass_per_length_kg_m))
    for indices, rigidity, density in (
        ([2, 8], axial, station.linear_density_kg_m),
        ([5, 11], station.gj_n_m2, polar),
    ):
        stiffness[np.ix_(indices, indices)] = rigidity / length * _ROD_GRADIENT
        mass[np.ix_(indices, indices)] = density * length * _ROD_MASS
    rotation = _principal_rotation(station.spine_angle_rad)
    return rotation @ stiffness @ rotation.T, rotation @ mass @ rotation.T


def tip_spatial_inertia(body: ComponentMassProperties) -> np.ndarray:
    """Kinetic energy of full COM inertia and v_com = v_tip + omega cross r."""
    # Column j is e_j cross r, avoiding an independent skew convention.
    com_motion = np.column_stack(
        (np.eye(3), np.cross(np.eye(3), body.center_of_mass_m).T)
    )
    result = body.mass_kg * com_motion.T @ com_motion
    result[3:, 3:] += body.inertia_at_com_kg_m2
    return result
