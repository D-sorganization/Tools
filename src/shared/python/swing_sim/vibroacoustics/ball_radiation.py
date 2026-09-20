"""Ball impact dipole acoustic radiation solver (IA-T5, #5074).

Solves direct acoustic radiation from ball compression and deformation during impact,
modeled as an acoustic dipole driven by the dynamic contact force history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ._boundary_radiation_records import AcousticMedium
from ._spectral_frames import finite_output
from ._waveform_contracts import owned_real_samples, real_samples
from .observer import ObserverLocation

logger = logging.getLogger(__name__)


def _real_matrix(array: object, name: str) -> np.ndarray:
    if isinstance(array, np.ndarray) and array.dtype == bool:
        raise ValueError(f"{name} cannot be boolean")
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 3D force history")
    if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(
        arr.dtype, np.integer
    ):
        raise ValueError(f"{name} must be numeric")
    float_arr = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(float_arr)):
        raise ValueError(f"{name} must contain only finite numbers")
    return float_arr


@dataclass(frozen=True)
class BallAcousticRadiation:
    """Acoustic dipole radiation solver for golf ball impact dynamics.

    Attributes:
        medium: Acoustic fluid medium properties.
    """

    medium: AcousticMedium = AcousticMedium()

    def solve_ball_pressure(
        self,
        observer: ObserverLocation,
        force_history_n: np.ndarray,
        time: np.ndarray,
        source_origin_m: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate radiated acoustic pressure from ball impact contact force history.

        Args:
            observer: Receiver location.
            force_history_n: (3, N_timesteps) 3D contact force vector history
                [Fx, Fy, Fz] in Newtons.
            time: (N_timesteps,) time vector in seconds.
            source_origin_m: (3,) contact point / COP coordinates (default: [0, 0, 0]).

        Returns:
            (N_timesteps,) radiated acoustic pressure in Pascals.
        """
        if not isinstance(observer, ObserverLocation):
            raise TypeError("observer must be an ObserverLocation")
        f_vec = _real_matrix(force_history_n, "force_history_n")
        t = real_samples(time)

        if f_vec.ndim != 2 or f_vec.shape[0] != 3:
            raise ValueError(
                "force_history_n must be a (3, N_timesteps) 3D force history"
            )
        if t.ndim != 1 or f_vec.shape[1] != t.size:
            raise ValueError("time must match force history columns")
        if t.size < 2:
            raise ValueError("time sequence must contain at least 2 points")

        origin = (
            np.zeros(3)
            if source_origin_m is None
            else np.asarray(source_origin_m, dtype=np.float64)
        )
        r_vec = observer.coordinates_m - origin
        dist = float(np.linalg.norm(r_vec))
        if dist < 1e-6:
            raise ValueError("Observer coincides with impact source location")

        unit_r = r_vec / dist
        c0 = self.medium.sound_speed_mps
        tau = dist / c0

        dt = float(t[1] - t[0])
        t0 = float(t[0])

        # Numerical time derivative of force vector: dF/dt (central differences)
        df_dt = np.gradient(f_vec, dt, axis=1)

        # Dot product with observer direction:
        # F_r(t) = unit_r . F(t), dF_r/dt = unit_r . dF/dt
        f_radial = unit_r @ f_vec
        df_radial = unit_r @ df_dt

        # Retarded time sampling
        t_source = t - tau
        source_indices = (t_source - t0) / dt
        valid_mask = (source_indices >= 0.0) & (source_indices < (t.size - 1))

        pressure = np.zeros(t.size, dtype=np.float64)
        if not np.any(valid_mask):
            return owned_real_samples(pressure)

        idx_floor = np.floor(source_indices[valid_mask]).astype(np.int64)
        fraction = source_indices[valid_mask] - idx_floor

        # Linear interpolation
        f_ret = (1.0 - fraction) * f_radial[idx_floor] + fraction * f_radial[
            idx_floor + 1
        ]
        df_ret = (1.0 - fraction) * df_radial[idx_floor] + fraction * df_radial[
            idx_floor + 1
        ]

        # Dipole formula: p(r, t) = (1 / 4pi*c*r) * dF/dt + (1 / 4pi*r^2) * F
        far_field = (1.0 / (4.0 * np.pi * c0 * dist)) * df_ret
        near_field = (1.0 / (4.0 * np.pi * (dist**2))) * f_ret

        pressure[valid_mask] = far_field + near_field
        return owned_real_samples(finite_output(pressure))


__all__ = [
    "BallAcousticRadiation",
]
