"""Calibrated waveform measurement ingestion (IA-T5, #5074).

Imports pressure/waveform recordings with calibration units and provenance,
provides raw-data content hashes, clipping and bandwidth checks, and
cross-correlation time alignment.  Synthesized sources are structurally
distinct and refused as measurements.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import signal as _signal

from ._waveform_contracts import owned_real_samples as _owned_real_samples
from ._waveform_contracts import real_samples as _real_samples


class SourceKind(Enum):
    """Provenance kind of a waveform recording."""

    MEASURED = "measured"
    SYNTHESIZED = "synthesized"


class SynthesizedSourceError(ValueError):
    """Raised when a synthesized signal is used as a measurement."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real scalar")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class WaveformRecording:
    """One waveform with caller-declared calibration and source metadata.

    Attributes:
        samples: Owned, immutable, one-dimensional finite real sample sequence.
        sample_rate_hz: Uniform sampling rate in hertz.
        unit: Calibration unit of the samples, e.g. ``Pa`` or ``N``.
        calibration_id: Identity of the calibration record.
        sensitivity_per_unit: Declared sensitivity, finite and nonzero; not
            applied to samples by this record or the spectral estimators.
        source_id: Identity of the physical or synthetic source.
        source_kind: Whether the source was measured or synthesized.
    """

    samples: np.ndarray
    sample_rate_hz: float
    unit: str
    calibration_id: str
    sensitivity_per_unit: float
    source_id: str
    source_kind: SourceKind = SourceKind.MEASURED

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _owned_real_samples(self.samples))
        rate = _finite(self.sample_rate_hz, "sample_rate_hz")
        if rate <= 0.0:
            raise ValueError("sample_rate_hz must be > 0")
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "unit", _label(self.unit, "unit"))
        object.__setattr__(
            self, "calibration_id", _label(self.calibration_id, "calibration_id")
        )
        sensitivity = _finite(self.sensitivity_per_unit, "sensitivity_per_unit")
        if sensitivity == 0.0:
            raise ValueError("sensitivity_per_unit must be nonzero")
        object.__setattr__(self, "sensitivity_per_unit", sensitivity)
        object.__setattr__(self, "source_id", _label(self.source_id, "source_id"))
        if not isinstance(self.source_kind, SourceKind):
            raise ValueError("source_kind must be a SourceKind")


def raw_data_hash(recording: WaveformRecording) -> str:
    """Return the legacy hash of sample bytes and selected calibration metadata.

    The hash binds the sample bytes (little-endian float64) to the sample
    rate, unit and calibration ID. Sensitivity and acquisition/source identity
    are not bound; this legacy digest does not authenticate calibration. Its
    existing byte convention is retained for archived-result compatibility.
    """
    if not isinstance(recording, WaveformRecording):
        raise TypeError("recording must be a WaveformRecording")
    header = json.dumps(
        {
            "sample_rate_hz": recording.sample_rate_hz,
            "unit": recording.unit,
            "calibration_id": recording.calibration_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(np.asarray(recording.samples, dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ClippingReport:
    """Outcome of a clipping check over one recording."""

    clipped_samples: int
    total_samples: int
    fraction: float
    clipped: bool


def clipping_fraction(recording: WaveformRecording, *, rail: float) -> ClippingReport:
    """Report the fraction of samples at or beyond ``rail`` in magnitude.

    Raises:
        ValueError: If ``rail`` is not finite and strictly positive.
    """
    if not isinstance(recording, WaveformRecording):
        raise TypeError("recording must be a WaveformRecording")
    limit = _finite(rail, "rail")
    if limit <= 0.0:
        raise ValueError("rail must be > 0")
    clipped = int(np.count_nonzero(np.abs(recording.samples) >= limit))
    total = int(recording.samples.size)
    fraction = clipped / total
    return ClippingReport(
        clipped_samples=clipped,
        total_samples=total,
        fraction=fraction,
        clipped=clipped > 0,
    )


@dataclass(frozen=True)
class BandwidthReport:
    """Outcome of a bandwidth adequacy check for one recording."""

    sample_rate_hz: float
    nyquist_hz: float
    max_expected_hz: float
    adequate: bool


def bandwidth_report(
    recording: WaveformRecording, *, max_expected_hz: float
) -> BandwidthReport:
    """Check that the Nyquist band covers ``max_expected_hz``.

    Raises:
        ValueError: If ``max_expected_hz`` is not finite and positive.
    """
    if not isinstance(recording, WaveformRecording):
        raise TypeError("recording must be a WaveformRecording")
    expected = _finite(max_expected_hz, "max_expected_hz")
    if expected <= 0.0:
        raise ValueError("max_expected_hz must be > 0")
    nyquist = recording.sample_rate_hz / 2.0
    return BandwidthReport(
        sample_rate_hz=recording.sample_rate_hz,
        nyquist_hz=nyquist,
        max_expected_hz=expected,
        adequate=expected < nyquist,
    )


def align_time_shift(reference: np.ndarray, delayed: np.ndarray) -> int:
    """Return the signed lag maximizing unnormalized linear cross-correlation.

    Positive ``k`` describes a delay: ``delayed[n] == reference[n - k]``
    on their overlap. Outside the finite records samples are zero, not wrapped.
    This peak heuristic does not establish synchronization or uncertainty.

    Raises:
        ValueError: If the arrays differ in length or are not finite
            real one-dimensional sequences of equal size, or the positive
            correlation peak is absent, tied or nonfinite.
    """
    reference_array = _real_samples(reference)
    delayed_array = _real_samples(delayed)
    if reference_array.shape != delayed_array.shape or reference_array.ndim != 1:
        raise ValueError("inputs must be same length, one-dimensional arrays")
    with np.errstate(over="ignore", invalid="ignore"):
        correlation = _signal.correlate(delayed_array, reference_array, mode="full")
    if not np.all(np.isfinite(correlation)):
        raise ValueError("correlation must be finite")
    peak = np.max(correlation)
    if peak <= 0 or np.count_nonzero(correlation == peak) != 1:
        raise ValueError("correlation needs one positive, unambiguous peak")
    return int(np.argmax(correlation) - (reference_array.size - 1))


def as_measured(recording: WaveformRecording) -> WaveformRecording:
    """Return ``recording`` only when its source kind is measured.

    Raises:
        SynthesizedSourceError: If the recording was synthesized.
            Callers must never present synthesized tones as predicted
            acoustics.
    """
    if not isinstance(recording, WaveformRecording):
        raise TypeError("recording must be a WaveformRecording")
    if recording.source_kind is not SourceKind.MEASURED:
        raise SynthesizedSourceError(
            f"recording {recording.source_id!r} is synthesized and must not "
            "be treated as a measurement"
        )
    return recording
