"""Immutable declared acquisition identities and a uniform sample clock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._calibration_primitives import (
    finite_scalar,
    identity_text,
    instance,
    sha256_reference,
)
from ._waveform_contracts import owned_real_samples
from .measurement import SourceKind


@dataclass(frozen=True)
class AcquisitionChannel:
    """Identified channel and sensor/ADC chain; setup digest is a declaration."""

    sensor_chain_id: str
    channel_id: str
    unit: str
    setup_sha256: str

    def __post_init__(self) -> None:
        for name in ("sensor_chain_id", "channel_id", "unit"):
            object.__setattr__(self, name, identity_text(getattr(self, name), name))
        sha256_reference(self.setup_sha256, "setup_sha256")


@dataclass(frozen=True)
class AcquisitionClock:
    """Uniform times in an identified clock; no synchronization/phase inference.

    Times are seconds in this clock, not an implicitly converted UTC timestamp.
    The evidence digest must be retained with any clock uncertainty/traceability.
    """

    clock_id: str
    first_sample_time_s: float
    sample_rate_hz: float
    timing_evidence_sha256: str

    def __post_init__(self) -> None:
        identity_text(self.clock_id, "clock_id")
        sha256_reference(self.timing_evidence_sha256, "timing_evidence_sha256")
        for name in ("first_sample_time_s", "sample_rate_hz"):
            object.__setattr__(self, name, finite_scalar(getattr(self, name), name))
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

    def last_sample_time_s(self, count: int) -> float:
        """Refuse unrepresentable time spacing rather than alias distinct samples."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("sample count must be a positive integer")
        first = self.first_sample_time_s
        last = finite_scalar(
            first + (count - 1) / self.sample_rate_hz, "last sample time"
        )
        spacing = float(np.spacing(max(abs(first), abs(last))))
        if count > 1 and (last <= first or 1 / self.sample_rate_hz < spacing):
            raise ValueError("sample clock has insufficient floating-point resolution")
        return last


@dataclass(frozen=True)
class WaveformProvenance:
    """Declared source kind and raw-file identity; neither authenticates a file."""

    source_id: str
    source_kind: SourceKind
    raw_file_sha256: str

    def __post_init__(self) -> None:
        identity_text(self.source_id, "source_id")
        instance(self.source_kind, SourceKind, "source_kind")
        sha256_reference(self.raw_file_sha256, "raw_file_sha256")


@dataclass(frozen=True)
class RawWaveform:
    """Owned raw indications; separate from already interpreted legacy samples."""

    samples: np.ndarray
    channel: AcquisitionChannel
    clock: AcquisitionClock
    provenance: WaveformProvenance

    def __post_init__(self) -> None:
        instance(self.channel, AcquisitionChannel, "channel")
        instance(self.clock, AcquisitionClock, "clock")
        instance(self.provenance, WaveformProvenance, "provenance")
        object.__setattr__(self, "samples", owned_real_samples(self.samples))
        self.clock.last_sample_time_s(self.size)

    @property
    def size(self) -> int:
        """Return the number of raw indications."""
        return int(self.samples.size)

    @property
    def unit(self) -> str:
        """Return the explicitly declared indication unit."""
        return self.channel.unit

    @property
    def source_kind(self) -> SourceKind:
        """Return provenance without elevating its authority."""
        return self.provenance.source_kind

    @property
    def time_extent_s(self) -> tuple[float, float]:
        """Return first/last sample times in the declared clock."""
        return self.clock.first_sample_time_s, self.clock.last_sample_time_s(self.size)


@dataclass(frozen=True)
class IndependentSampleUncertainty:
    """Declared standard uncertainties for mutually independent indications.

    Also assumes independence from calibration parameters. It is an explicit
    model, not a default or an estimate from the signal's physical amplitude.
    """

    standard_uncertainty: np.ndarray

    def __post_init__(self) -> None:
        values = owned_real_samples(self.standard_uncertainty)
        if np.any(values < 0):
            raise ValueError("standard uncertainty must be nonnegative")
        object.__setattr__(self, "standard_uncertainty", values)


__all__ = ()
