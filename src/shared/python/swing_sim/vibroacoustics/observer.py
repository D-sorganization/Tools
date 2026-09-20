"""Observer locations, microphone arrays, and held-out comparison (IA-T5, #5074).

Provides geometric observer placement, standard golfer-ear and field microphone
arrays, and validation comparison between predicted and measured acoustic waveforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ._waveform_contracts import owned_real_samples, real_samples

logger = logging.getLogger(__name__)


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real scalar")
    num = float(value)
    if not np.isfinite(num):
        raise ValueError(f"{name} must be finite")
    return num


@dataclass(frozen=True)
class ObserverLocation:
    """An acoustic field evaluation or microphone receiver position.

    Attributes:
        name: Nonempty identifier for the observer.
        coordinates_m: (3,) Cartesian position vector in meters [x, y, z].
        channel_id: Optional acquisition channel label.
    """

    name: str
    coordinates_m: np.ndarray
    channel_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Observer name must be a nonempty string")
        coords = owned_real_samples(self.coordinates_m)
        if coords.shape != (3,):
            raise ValueError("coordinates_m must have shape (3,)")
        object.__setattr__(self, "coordinates_m", coords)

    def distance_to(self, origin_m: np.ndarray | None = None) -> float:
        """Euclidean distance from origin_m (default: [0, 0, 0])."""
        origin = (
            np.zeros(3) if origin_m is None else np.asarray(origin_m, dtype=np.float64)
        )
        return float(np.linalg.norm(self.coordinates_m - origin))

    def spherical_coordinates(
        self, origin_m: np.ndarray | None = None
    ) -> tuple[float, float, float]:
        """Return (r, theta, phi) in meters and radians.

        theta is polar angle from +z axis [0, pi], phi is azimuth in xy-plane [-pi, pi].
        """
        origin = (
            np.zeros(3) if origin_m is None else np.asarray(origin_m, dtype=np.float64)
        )
        delta = self.coordinates_m - origin
        r = float(np.linalg.norm(delta))
        if r < 1e-12:
            return 0.0, 0.0, 0.0
        theta = float(np.arccos(np.clip(delta[2] / r, -1.0, 1.0)))
        phi = float(np.arctan2(delta[1], delta[0]))
        return r, theta, phi


@dataclass(frozen=True)
class MicrophoneArray:
    """Collection of observer locations representing a multi-microphone setup."""

    observers: tuple[ObserverLocation, ...]

    def __post_init__(self) -> None:
        if not self.observers:
            raise ValueError("MicrophoneArray requires at least one observer")
        for obs in self.observers:
            if not isinstance(obs, ObserverLocation):
                raise TypeError("observers must contain ObserverLocation instances")

    def get_observer(self, name: str) -> ObserverLocation:
        """Find an observer by name."""
        for obs in self.observers:
            if obs.name == name:
                return obs
        raise KeyError(f"Observer {name!r} not found in array")

    @classmethod
    def standard_golfer_and_field_microphones(cls) -> MicrophoneArray:
        """Standard golf vibroacoustic microphone layout relative to impact origin."""
        return cls(
            observers=(
                ObserverLocation(
                    name="golfer_ear", coordinates_m=np.array([0.0, -0.3, 1.6])
                ),
                ObserverLocation(
                    name="front_1m", coordinates_m=np.array([1.0, 0.0, 0.3])
                ),
                ObserverLocation(
                    name="off_axis_45deg", coordinates_m=np.array([0.707, 0.707, 0.3])
                ),
                ObserverLocation(
                    name="rear_2m", coordinates_m=np.array([-2.0, 0.0, 0.5])
                ),
            )
        )


@dataclass(frozen=True)
class HeldOutObserverComparison:
    """Comparison metrics between predicted and measured acoustic pressure signals."""

    observer: ObserverLocation
    rms_error_pa: float
    relative_l2_error: float
    peak_spl_error_db: float
    cross_correlation_max: float
    spectral_error_db: float

    @classmethod
    def compare_waveforms(
        cls,
        predicted_pressure: np.ndarray,
        target_pressure: np.ndarray,
        observer: ObserverLocation,
        sample_rate_hz: float,
        p0: float = 20e-6,
    ) -> HeldOutObserverComparison:
        """Compute comprehensive comparison metrics between two acoustic signals."""
        pred = real_samples(predicted_pressure)
        targ = real_samples(target_pressure)
        if pred.shape != targ.shape or pred.ndim != 1:
            raise ValueError(
                "predicted and target pressure must be 1D arrays of identical length"
            )
        if pred.size == 0:
            raise ValueError("pressure arrays must not be empty")

        diff = pred - targ
        rms_err = float(np.sqrt(np.mean(diff**2)))
        norm_targ = float(np.linalg.norm(targ))
        rel_l2 = (
            float(np.linalg.norm(diff) / norm_targ)
            if norm_targ > 1e-12
            else float(np.linalg.norm(diff))
        )

        # Peak SPL comparison
        peak_pred = float(np.max(np.abs(pred)))
        peak_targ = float(np.max(np.abs(targ)))
        spl_pred = 20.0 * np.log10(max(peak_pred, 1e-12) / p0)
        spl_targ = 20.0 * np.log10(max(peak_targ, 1e-12) / p0)
        peak_spl_err = float(abs(spl_pred - spl_targ))

        # Normalized cross correlation peak
        norm_factor = np.linalg.norm(pred) * np.linalg.norm(targ)
        if norm_factor > 1e-12:
            xcorr = np.correlate(pred, targ, mode="full") / norm_factor
            max_xcorr = float(np.max(xcorr))
        else:
            max_xcorr = 1.0 if np.allclose(pred, targ) else 0.0

        # Spectral difference in dB
        fft_pred = np.abs(np.fft.rfft(pred))
        fft_targ = np.abs(np.fft.rfft(targ))
        eps = 1e-12
        db_diff = 20.0 * np.log10((fft_pred + eps) / (fft_targ + eps))
        spectral_err = float(np.mean(np.abs(db_diff)))

        return cls(
            observer=observer,
            rms_error_pa=rms_err,
            relative_l2_error=rel_l2,
            peak_spl_error_db=peak_spl_err,
            cross_correlation_max=max_xcorr,
            spectral_error_db=spectral_err,
        )


__all__ = [
    "HeldOutObserverComparison",
    "MicrophoneArray",
    "ObserverLocation",
]
