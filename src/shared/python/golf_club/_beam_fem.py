"""Shared private Euler-Bernoulli assembly; SI translation/slope coordinates."""

from __future__ import annotations

import numpy as np

from .shaft_profile import ShaftProfile


def assemble_bending_axis(
    profile: ShaftProfile, element_count: int, stiffness_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return unconstrained midpoint-property stiffness and consistent mass."""
    element_length = profile.flexible_length_m / element_count
    degrees = 2 * (element_count + 1)
    stiffness: np.ndarray = np.zeros((degrees, degrees), dtype=float)
    mass: np.ndarray = np.zeros((degrees, degrees), dtype=float)
    start = profile.butt_trim_m
    for element in range(element_count):
        midpoint = start + (element + 0.5) * element_length
        station = profile.station_at(midpoint)
        local_stiffness = beam_stiffness(
            float(getattr(station, stiffness_name)), element_length
        )
        local_mass = beam_mass(station.linear_density_kg_m, element_length)
        indices = np.array(
            [2 * element, 2 * element + 1, 2 * element + 2, 2 * element + 3]
        )
        stiffness[np.ix_(indices, indices)] += local_stiffness
        mass[np.ix_(indices, indices)] += local_mass
    return stiffness, mass


def beam_stiffness(ei_n_m2: float, length_m: float) -> np.ndarray:
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


def beam_mass(linear_density_kg_m: float, length_m: float) -> np.ndarray:
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


def generalized_eigenvalues(stiffness: np.ndarray, mass: np.ndarray) -> np.ndarray:
    factor = np.linalg.cholesky(mass)
    left_solved = np.linalg.solve(factor, stiffness)
    transformed = np.linalg.solve(factor, left_solved.T).T
    symmetric = 0.5 * (transformed + transformed.T)
    eigenvalues: np.ndarray = np.asarray(np.linalg.eigvalsh(symmetric))
    return eigenvalues
