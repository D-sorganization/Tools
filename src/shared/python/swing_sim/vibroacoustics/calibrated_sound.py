"""Calibrated acoustic pressure recordings and timebase synchronization (IA-T5, #5074).

Binds verified CalibratedWaveform records with physical observer geometry,
authenticating acoustic pressure units ('Pa'), preserving exact SHA-256 provenance,
and providing phase-sensitive timebase synchronization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .measurement import (
    SourceKind,
    SynthesizedSourceError,
    align_time_shift,
)
from .observer import ObserverLocation
from .waveform_calibration import CalibratedWaveform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibratedSoundRecording:
    """Verified acoustic pressure recording binding CalibratedWaveform &
    ObserverLocation.

    Attributes:
        waveform: Calibrated waveform record with exact identity and uncertainty model.
        observer: 3D receiver location in space.
    """

    waveform: CalibratedWaveform
    observer: ObserverLocation

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, CalibratedWaveform):
            raise TypeError("waveform must be a CalibratedWaveform")
        if not isinstance(self.observer, ObserverLocation):
            raise TypeError("observer must be an ObserverLocation")
        if self.waveform.unit != "Pa":
            unit = self.waveform.unit
            raise ValueError(
                f"CalibratedSoundRecording unit must be 'Pa', got {unit!r}"
            )

    @property
    def unit(self) -> str:
        """Physical unit of the pressure waveform ('Pa')."""
        return self.waveform.unit

    @property
    def sample_rate_hz(self) -> float:
        """Acquisition sampling rate in Hz."""
        return self.waveform.raw.clock.sample_rate_hz

    @property
    def samples(self) -> np.ndarray:
        """Converted acoustic pressure samples in Pascals."""
        return self.waveform.values

    @property
    def identity_sha256(self) -> str:
        """Canonical SHA-256 identity binding provenance and sample bytes."""
        return self.waveform.identity_sha256

    @property
    def source_kind(self) -> SourceKind:
        """Provenance kind (MEASURED or SYNTHESIZED)."""
        return self.waveform.source_kind

    def as_measured(self) -> CalibratedSoundRecording:
        """Return self if measured; raise SynthesizedSourceError if synthesized.

        Never label synthesized tones as predicted acoustics.
        """
        if self.source_kind is not SourceKind.MEASURED:
            raise SynthesizedSourceError(
                f"recording at observer {self.observer.name!r} is synthesized "
                "and must not be treated as a physical measurement"
            )
        return self


def synchronize_timebases(
    reference: CalibratedSoundRecording,
    observed: CalibratedSoundRecording,
    distance_m: float,
    sound_speed_mps: float = 343.2,
) -> tuple[int, float]:
    """Phase-sensitive cross-correlation synchronization across a known distance.

    Accounts for acoustic propagation time delay tau = distance / sound_speed.
    Returns (residual_lag_samples, propagation_delay_seconds).

    Args:
        reference: Reference emission/sensor recording.
        observed: Receiver recording at distance_m.
        distance_m: Distance between source and observer in meters.
        sound_speed_mps: Ambient sound speed in m/s.

    Returns:
        (residual_lag_samples, propagation_delay_seconds)
    """
    if not isinstance(reference, CalibratedSoundRecording) or not isinstance(
        observed, CalibratedSoundRecording
    ):
        raise TypeError("inputs must be CalibratedSoundRecording instances")
    if reference.sample_rate_hz != observed.sample_rate_hz:
        raise ValueError(
            "Sampling rates must be identical for timebase synchronization"
        )
    if distance_m < 0.0:
        raise ValueError("distance_m must be nonnegative")
    if sound_speed_mps <= 0.0:
        raise ValueError("sound_speed_mps must be positive")

    fs = reference.sample_rate_hz
    propagation_delay_s = distance_m / sound_speed_mps
    propagation_samples = int(np.round(propagation_delay_s * fs))

    # Total observed delay from linear cross-correlation peak
    total_lag = align_time_shift(reference.samples, observed.samples)

    # Residual synchronization offset after accounting for acoustic propagation
    residual_lag = total_lag - propagation_samples
    return residual_lag, propagation_delay_s


__all__ = [
    "CalibratedSoundRecording",
    "synchronize_timebases",
]
