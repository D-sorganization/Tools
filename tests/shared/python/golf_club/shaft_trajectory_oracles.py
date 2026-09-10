"""Independent scalar rod continuum and semi-discrete time references."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq


@dataclass(frozen=True)
class AxialRodReference:
    """Synthetic SI rod with a spring/inertance root and massive free tip.

    Parameters match the existing synthetic fixture independently of its
    assembled production matrices. No measurements or equipment fit is claimed.
    """

    length_m: float = 1.0
    rigidity_n: float = 1000.0
    density_kg_m: float = 0.2
    tip_mass_kg: float = 0.1
    root_stiffness_n_m: float = 400.0
    root_inertance_kg: float = 0.03
    amplitude_m: float = 0.001

    def shape(
        self, omega: float, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        wave = omega * np.sqrt(self.density_kg_m / self.rigidity_n)
        support = self.root_stiffness_n_m - self.root_inertance_kg * omega**2
        ratio = support / (self.rigidity_n * wave)
        phase = wave * positions
        shape = np.cos(phase) + ratio * np.sin(phase)
        slope = wave * (-np.sin(phase) + ratio * np.cos(phase))
        return shape, slope

    def characteristic(self, omega: float) -> float:
        shape, slope = self.shape(omega, np.array([self.length_m]))
        return float(
            self.rigidity_n * slope[0] - self.tip_mass_kg * omega**2 * shape[0]
        )

    @property
    def omega_rad_s(self) -> float:
        omega = float(brentq(self.characteristic, 5, 60, xtol=1e-13))
        assert abs(self.characteristic(omega)) < 1e-10
        return omega

    def positions(self, count: int) -> np.ndarray:
        return np.linspace(0, self.length_m, count + 1)

    def continuum_state(self, count: int, time_s: float) -> np.ndarray:
        omega = self.omega_rad_s
        shape, _ = self.shape(omega, self.positions(count))
        motion = self.amplitude_m * shape
        return np.r_[
            motion * np.cos(omega * time_s), -omega * motion * np.sin(omega * time_s)
        ]

    def matrices(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        mass, stiffness = (
            np.zeros((count + 1, count + 1)),
            np.zeros((count + 1, count + 1)),
        )
        element_length = self.length_m / count
        for index in range(count):
            rows = slice(index, index + 2)
            mass[rows, rows] += (
                self.density_kg_m * element_length / 6 * np.array([[2, 1], [1, 2]])
            )
            stiffness[rows, rows] += (
                self.rigidity_n / element_length * np.array([[1, -1], [-1, 1]])
            )
        mass[0, 0] += self.root_inertance_kg
        mass[-1, -1] += self.tip_mass_kg
        stiffness[0, 0] += self.root_stiffness_n_m
        return mass, stiffness

    def discrete_state(self, count: int, time_s: float) -> np.ndarray:
        mass, stiffness = self.matrices(count)
        generator = np.block(
            [
                [np.zeros_like(mass), np.eye(count + 1)],
                [-np.linalg.solve(mass, stiffness), np.zeros_like(mass)],
            ]
        )
        return np.asarray(expm(time_s * generator) @ self.continuum_state(count, 0))

    def discrete_energy(self, count: int, state: np.ndarray) -> float:
        mass, stiffness = self.matrices(count)
        displacement, velocity = state[: count + 1], state[count + 1 :]
        return float(
            (displacement @ stiffness @ displacement + velocity @ mass @ velocity) / 2
        )

    def continuum_energy(self, time_s: float) -> float:
        omega = self.omega_rad_s

        def density(position: float) -> float:
            shape, slope = self.shape(omega, np.array([position]))
            return float(
                self.rigidity_n * (slope[0] * np.cos(omega * time_s)) ** 2
                + self.density_kg_m * (omega * shape[0] * np.sin(omega * time_s)) ** 2
            )

        ends = self.continuum_state(1, time_s)
        boundary = (
            self.root_stiffness_n_m * ends[0] ** 2
            + self.root_inertance_kg * ends[2] ** 2
            + self.tip_mass_kg * ends[3] ** 2
        )
        return float(
            (
                self.amplitude_m**2 * quad(density, 0, self.length_m, epsabs=1e-11)[0]
                + boundary
            )
            / 2
        )

    def error(self, count: int, actual: np.ndarray, reference: np.ndarray) -> float:
        scales = np.r_[
            np.full(count + 1, self.amplitude_m),
            np.full(count + 1, self.amplitude_m * self.omega_rad_s),
        ]
        return float(np.max(abs(actual - reference) / scales))
