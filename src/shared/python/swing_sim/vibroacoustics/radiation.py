"""Transient vibroacoustic radiation solver and modal transfer (IA-T5, #5074).

Solves the retarded-time Rayleigh surface integral for baffled planar/curved radiators
and modal sound radiation into an acoustic half-space or free field.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._boundary_radiation_records import AcousticMedium, RadiatingSurfaceMesh
from ._spectral_frames import finite_output
from ._waveform_contracts import owned_real_samples, real_samples
from .observer import ObserverLocation

logger = logging.getLogger(__name__)


def _real_matrix(array: object, name: str) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == bool:
        raise ValueError(f"{name} cannot be boolean")
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(
        arr.dtype, np.integer
    ):
        raise ValueError(f"{name} must be numeric")
    float_arr = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(float_arr)):
        raise ValueError(f"{name} must contain only finite numbers")
    return float_arr


@dataclass(frozen=True)
class TransientRadiationSolver:
    """Rayleigh surface integral transient acoustic radiation solver.

    Attributes:
        mesh: Radiating surface mesh containing discretized boundary elements.
        medium: Acoustic fluid medium properties.
        baffled: If True, radiates into half-space (factor rho0 / 2pi);
            if False, unbaffled free-field dipole (factor rho0 / 4pi).
    """

    mesh: RadiatingSurfaceMesh
    medium: AcousticMedium = AcousticMedium()
    baffled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, RadiatingSurfaceMesh):
            raise TypeError("mesh must be a RadiatingSurfaceMesh")
        if not isinstance(self.medium, AcousticMedium):
            raise TypeError("medium must be an AcousticMedium")

    def solve_transient_pressure(
        self,
        observer: ObserverLocation,
        acceleration_matrix: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        """Calculate radiated acoustic pressure history at observer.

        Args:
            observer: Receiver location in 3D space.
            acceleration_matrix: (N_elements, N_timesteps) normal surface
                acceleration history.
            time: (N_timesteps,) uniform time points in seconds.

        Returns:
            (N_timesteps,) acoustic pressure history in Pascals.
        """
        if not isinstance(observer, ObserverLocation):
            raise TypeError("observer must be an ObserverLocation")
        accel = _real_matrix(acceleration_matrix, "acceleration_matrix")
        t = real_samples(time)

        n_elem = len(self.mesh.elements)
        if accel.ndim != 2 or accel.shape[0] != n_elem:
            raise ValueError(
                f"acceleration_matrix must have shape ({n_elem}, N_timesteps)"
            )
        if t.ndim != 1 or accel.shape[1] != t.size:
            raise ValueError("time must be 1D with same length as acceleration columns")
        if t.size < 2:
            raise ValueError("time sequence must contain at least 2 points")

        dt = float(t[1] - t[0])
        if dt <= 0.0:
            raise ValueError("time points must be strictly increasing")
        t0 = float(t[0])

        c0 = self.medium.sound_speed_mps
        rho0 = self.medium.density_kg_per_m3
        scale = (rho0 / (2.0 * np.pi)) if self.baffled else (rho0 / (4.0 * np.pi))

        pressure = np.zeros(t.size, dtype=np.float64)

        for i, el in enumerate(self.mesh.elements):
            r_vec = observer.coordinates_m - el.centroid_m
            dist = float(np.linalg.norm(r_vec))
            if dist < 1e-6:
                raise ValueError("Observer coincides with radiating element surface")

            # Retarded arrival time
            tau = dist / c0
            # Target times at source: t_source = t - tau
            t_source = t - tau

            # Retarded acceleration using linear interpolation
            source_indices = (t_source - t0) / dt
            valid_mask = (source_indices >= 0.0) & (source_indices < (t.size - 1))

            if not np.any(valid_mask):
                continue

            idx_floor = np.floor(source_indices[valid_mask]).astype(np.int64)
            fraction = source_indices[valid_mask] - idx_floor

            a_ret = (1.0 - fraction) * accel[i, idx_floor] + fraction * accel[
                i, idx_floor + 1
            ]

            # Accumulate contribution: scale * (delta_S / dist) * a_n(t - r/c)
            factor = scale * (el.area_m2 / dist)
            pressure[valid_mask] += factor * a_ret

        return owned_real_samples(finite_output(pressure))


@dataclass(frozen=True)
class ModalRadiationTransfer:
    """Modal acoustic radiation transfer from structural vibration modes to observer."""

    mesh: RadiatingSurfaceMesh
    medium: AcousticMedium
    mode_shapes: tuple[Callable[[float, float], float], ...]
    mode_names: tuple[str, ...] = ()
    baffled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, RadiatingSurfaceMesh):
            raise TypeError("mesh must be a RadiatingSurfaceMesh")
        if not self.mode_shapes:
            raise ValueError("mode_shapes cannot be empty")

    def solve_modal_pressure(
        self,
        observer: ObserverLocation,
        modal_accelerations: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        """Compute radiated acoustic pressure from modal acceleration coordinates.

        Args:
            observer: Receiver location.
            modal_accelerations: (M_modes, N_timesteps) modal acceleration history.
            time: (N_timesteps,) uniform time points.
        """
        q_ddot = _real_matrix(modal_accelerations, "modal_accelerations")
        t = real_samples(time)
        m_modes = len(self.mode_shapes)

        if q_ddot.ndim != 2 or q_ddot.shape[0] != m_modes:
            raise ValueError(
                f"modal_accelerations must have shape ({m_modes}, N_timesteps)"
            )
        if t.ndim != 1 or q_ddot.shape[1] != t.size:
            raise ValueError("time must match modal_accelerations length")

        n_elem = len(self.mesh.elements)
        # Synthesize normal acceleration field from mode shapes:
        # a_n(x, y, t) = sum_m phi_m(x, y) * q_ddot_m(t)
        accel_matrix = np.zeros((n_elem, t.size), dtype=np.float64)

        for i, el in enumerate(self.mesh.elements):
            x, y = float(el.centroid_m[0]), float(el.centroid_m[1])
            phi_vals = np.array([shape_fn(x, y) for shape_fn in self.mode_shapes])
            accel_matrix[i, :] = phi_vals @ q_ddot

        solver = TransientRadiationSolver(
            mesh=self.mesh, medium=self.medium, baffled=self.baffled
        )
        return solver.solve_transient_pressure(observer, accel_matrix, t)

    def modal_radiation_impedance(
        self,
        observer: ObserverLocation,
        angular_frequency_rad_per_s: float,
        mode_index: int,
    ) -> complex:
        """Complex frequency-domain transfer impedance Z_rad = p_obs / q_dot."""
        if mode_index < 0 or mode_index >= len(self.mode_shapes):
            raise IndexError("mode_index out of range")
        omega = float(angular_frequency_rad_per_s)
        k = omega / self.medium.sound_speed_mps
        rho0 = self.medium.density_kg_per_m3
        scale = (rho0 / (2.0 * np.pi)) if self.baffled else (rho0 / (4.0 * np.pi))

        shape_fn = self.mode_shapes[mode_index]
        z_rad = 0.0 + 0.0j

        for el in self.mesh.elements:
            dist = observer.distance_to(el.centroid_m)
            x, y = float(el.centroid_m[0]), float(el.centroid_m[1])
            phi = float(shape_fn(x, y))
            # Harmonic factor: i * omega * e^(-i * k * r) / r
            greens = np.exp(-1j * k * dist) / dist
            z_rad += scale * el.area_m2 * phi * (1j * omega) * greens

        return complex(z_rad)


__all__ = [
    "ModalRadiationTransfer",
    "TransientRadiationSolver",
]
