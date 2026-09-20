"""Face and hosel structural modes for clubhead impact dynamics.

Models high-frequency clubhead vibration modes (face trampoline, hosel bending,
and hosel torsion) and their work-conjugate coupling to the moving contact point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...golf_club._grip_contracts import finite_array
from ...golf_club._validation import require_finite_float, require_identifier


@dataclass(frozen=True)
class FaceHoselMode:
    """A single structural vibration mode of the clubhead face or hosel."""

    name: str
    modal_mass_kg: float
    modal_stiffness_n_per_m: float
    modal_damping_n_s_per_m: float
    shape_evaluator: Callable[[float, float], float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_identifier(self.name, "mode name"))
        object.__setattr__(
            self,
            "modal_mass_kg",
            require_finite_float(self.modal_mass_kg, "modal_mass_kg", positive=True),
        )
        object.__setattr__(
            self,
            "modal_stiffness_n_per_m",
            require_finite_float(
                self.modal_stiffness_n_per_m,
                "modal_stiffness_n_per_m",
                positive=True,
            ),
        )
        damping = require_finite_float(
            self.modal_damping_n_s_per_m, "modal_damping_n_s_per_m"
        )
        if damping < 0.0:
            raise ValueError("modal damping must be nonnegative")
        object.__setattr__(self, "modal_damping_n_s_per_m", damping)

    @property
    def natural_frequency_rad_per_s(self) -> float:
        return float(np.sqrt(self.modal_stiffness_n_per_m / self.modal_mass_kg))

    @property
    def natural_frequency_hz(self) -> float:
        return self.natural_frequency_rad_per_s / (2.0 * np.pi)

    @property
    def damping_ratio(self) -> float:
        critical = 2.0 * np.sqrt(self.modal_mass_kg * self.modal_stiffness_n_per_m)
        return float(self.modal_damping_n_s_per_m / critical)

    def shape_at(self, face_x_m: float, face_y_m: float) -> float:
        """Evaluate mode shape amplitude Phi(x, y) at face coordinates."""
        return float(self.shape_evaluator(face_x_m, face_y_m))


@dataclass(frozen=True)
class FaceHoselModalState:
    """State of all generalized coordinates eta and velocities d_eta/dt."""

    generalized_coordinates: np.ndarray
    generalized_velocities: np.ndarray

    def __post_init__(self) -> None:
        coords = finite_array(
            self.generalized_coordinates,
            (len(self.generalized_coordinates),),
            "generalized_coordinates",
        )
        vels = finite_array(
            self.generalized_velocities,
            (len(self.generalized_velocities),),
            "generalized_velocities",
        )
        if coords.shape != vels.shape:
            raise ValueError(
                "generalized coordinates and velocities must have the same dimension"
            )
        object.__setattr__(self, "generalized_coordinates", coords)
        object.__setattr__(self, "generalized_velocities", vels)

    @classmethod
    def zeros(cls, mode_count: int) -> FaceHoselModalState:
        return cls(
            generalized_coordinates=np.zeros(mode_count, dtype=np.float64),
            generalized_velocities=np.zeros(mode_count, dtype=np.float64),
        )


@dataclass(frozen=True)
class FaceHoselModalSystem:
    """Multi-mode structural system of clubhead face and hosel modes."""

    modes: tuple[FaceHoselMode, ...]

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("modes tuple cannot be empty")
        for m in self.modes:
            if not isinstance(m, FaceHoselMode):
                raise TypeError("each mode must be a FaceHoselMode")

    @property
    def mode_count(self) -> int:
        return len(self.modes)

    def generalized_forces(
        self, cop_coords_m: tuple[float, float], normal_force_n: float
    ) -> np.ndarray:
        """Compute generalized modal forces Q_m = Phi_m(cop) * F_n."""
        x, y = cop_coords_m
        return np.array(
            [m.shape_at(x, y) * normal_force_n for m in self.modes],
            dtype=np.float64,
        )

    def modal_accelerations(
        self, state: FaceHoselModalState, generalized_forces: np.ndarray
    ) -> np.ndarray:
        """Compute modal accelerations: (Q - C * v - K * q) / M."""
        accels = []
        q = state.generalized_coordinates
        v = state.generalized_velocities
        for i, m in enumerate(self.modes):
            net_force = (
                generalized_forces[i]
                - m.modal_damping_n_s_per_m * v[i]
                - m.modal_stiffness_n_per_m * q[i]
            )
            accels.append(net_force / m.modal_mass_kg)
        return np.array(accels, dtype=np.float64)

    def modal_deflection_m(
        self, state: FaceHoselModalState, cop_coords_m: tuple[float, float]
    ) -> float:
        """Total modal displacement u_modal(cop) = sum(Phi_m(cop) * q_m)."""
        x, y = cop_coords_m
        q = state.generalized_coordinates
        return float(sum(m.shape_at(x, y) * q[i] for i, m in enumerate(self.modes)))

    def modal_velocity_mps(
        self, state: FaceHoselModalState, cop_coords_m: tuple[float, float]
    ) -> float:
        """Total modal velocity v_modal(cop) = sum(Phi_m(cop) * v_m)."""
        x, y = cop_coords_m
        v = state.generalized_velocities
        return float(sum(m.shape_at(x, y) * v[i] for i, m in enumerate(self.modes)))

    def modal_energy_j(self, state: FaceHoselModalState) -> float:
        """Total stored modal energy: 0.5 * sum(M * v^2 + K * q^2)."""
        q = state.generalized_coordinates
        v = state.generalized_velocities
        return float(
            sum(
                0.5
                * (
                    m.modal_mass_kg * (v[i] ** 2)
                    + m.modal_stiffness_n_per_m * (q[i] ** 2)
                )
                for i, m in enumerate(self.modes)
            )
        )

    def modal_dissipation_power_w(self, state: FaceHoselModalState) -> float:
        """Instantaneous dissipated power: sum(C * v^2) >= 0."""
        v = state.generalized_velocities
        return float(
            sum(
                m.modal_damping_n_s_per_m * (v[i] ** 2)
                for i, m in enumerate(self.modes)
            )
        )


def standard_driver_face_hosel_modes() -> FaceHoselModalSystem:
    """Return standard 3-mode driver face/hosel modal system.

    1. Face trampoline mode (~4500 Hz):
       High-deflection membrane deformation at face center, fading toward edges.
    2. Hosel bending mode (~1800 Hz):
       First bending mode with heel-to-toe gradient across the face.
    3. Hosel torsion mode (~2200 Hz):
       Torsional mode with crown-to-sole gradient across the face.
    """
    face_w = 0.055
    face_h = 0.035

    def trampoline_shape(x: float, y: float) -> float:
        r2 = (x / face_w) ** 2 + (y / face_h) ** 2
        return float(max(0.0, 1.0 - r2) ** 2)

    def hosel_bending_shape(x: float, y: float) -> float:
        # Heel is at negative x; bending deflects toe relative to hosel
        return float((x + face_w) / (2.0 * face_w))

    def hosel_torsion_shape(x: float, y: float) -> float:
        # Sole-to-crown twist gradient
        return float(y / face_h)

    # Mode 1: Trampoline mode ~4500 Hz (omega ~ 28274 rad/s)
    # M = 0.035 kg -> K = 2.8e7 N/m, zeta = 0.02 -> C = 40 N*s/m
    trampoline = FaceHoselMode(
        name="face_trampoline",
        modal_mass_kg=0.035,
        modal_stiffness_n_per_m=2.8e7,
        modal_damping_n_s_per_m=40.0,
        shape_evaluator=trampoline_shape,
    )

    # Mode 2: Hosel bending mode ~1800 Hz (omega ~ 11310 rad/s)
    # M = 0.050 kg -> K = 6.4e6 N/m, zeta = 0.025 -> C = 28.3 N*s/m
    hosel_bending = FaceHoselMode(
        name="hosel_bending",
        modal_mass_kg=0.050,
        modal_stiffness_n_per_m=6.4e6,
        modal_damping_n_s_per_m=28.0,
        shape_evaluator=hosel_bending_shape,
    )

    # Mode 3: Hosel torsion mode ~2200 Hz (omega ~ 13823 rad/s)
    # M = 0.045 kg -> K = 8.6e6 N/m, zeta = 0.02 -> C = 24.9 N*s/m
    hosel_torsion = FaceHoselMode(
        name="hosel_torsion",
        modal_mass_kg=0.045,
        modal_stiffness_n_per_m=8.6e6,
        modal_damping_n_s_per_m=25.0,
        shape_evaluator=hosel_torsion_shape,
    )

    return FaceHoselModalSystem((trampoline, hosel_bending, hosel_torsion))


__all__ = (
    "FaceHoselMode",
    "FaceHoselModalState",
    "FaceHoselModalSystem",
    "standard_driver_face_hosel_modes",
)
