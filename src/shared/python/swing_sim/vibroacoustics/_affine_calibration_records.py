"""Declared affine conversion, metrology references and restricted validity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ._acquisition_records import RawWaveform
from ._calibration_primitives import (
    finite_scalar,
    identity_text,
    instance,
    interval,
    sha256_reference,
)
from ._frf_contracts import nonnegative_real
from .measurement import SourceKind


class UncertaintyPropagation(Enum):
    """Declared covariance law, both requiring calibration/input independence."""

    FIRST_ORDER = "first-order"
    EXACT_INDEPENDENT = "exact-independent-second-moments"


@dataclass(frozen=True)
class GainOffsetUncertainty:
    """Standard gain/offset uncertainty and correlation, with no coverage factor."""

    gain_standard_uncertainty: float
    offset_standard_uncertainty: float
    correlation: float

    def __post_init__(self) -> None:
        for name in ("gain_standard_uncertainty", "offset_standard_uncertainty"):
            object.__setattr__(self, name, nonnegative_real(getattr(self, name), name))
        rho = finite_scalar(self.correlation, "correlation")
        if abs(rho) > 1:
            raise ValueError("correlation must be in [-1, 1]")
        if (
            self.gain_standard_uncertainty == 0 or self.offset_standard_uncertainty == 0
        ) and rho != 0:
            raise ValueError(
                "zero standard uncertainty requires zero declared correlation"
            )
        object.__setattr__(self, "correlation", rho)


@dataclass(frozen=True)
class AffineCalibration:
    """Nominal y=g*x+b; units are (indication, output), gain may reverse polarity."""

    gain: float
    offset: float
    units: tuple[str, str]
    uncertainty: GainOffsetUncertainty | None

    def __post_init__(self) -> None:
        for name in ("gain", "offset"):
            object.__setattr__(self, name, finite_scalar(getattr(self, name), name))
        if self.gain == 0:
            raise ValueError("calibration gain must be nonzero")
        if not isinstance(self.units, tuple) or len(self.units) != 2:
            raise TypeError("units must be an indication/output tuple")
        for unit in self.units:
            identity_text(unit, "unit")
        if self.uncertainty is not None:
            instance(self.uncertainty, GainOffsetUncertainty, "uncertainty")


@dataclass(frozen=True)
class CalibrationDomain:
    """Declared raw range, valid flat-response band and same-clock time interval.

    Bounds are inclusive. The band does not assert that samples are bandlimited
    or replace actual frequency-dependent calibration/anti-aliasing evidence.
    """

    indication_range: tuple[float, float]
    frequency_band_hz: tuple[float, float]
    time_interval_s: tuple[float, float]
    clock_id: str

    def __post_init__(self) -> None:
        for name in ("indication_range", "frequency_band_hz", "time_interval_s"):
            object.__setattr__(self, name, interval(getattr(self, name), name))
        if self.frequency_band_hz[0] < 0:
            raise ValueError("frequency band must be nonnegative")
        identity_text(self.clock_id, "clock_id")


@dataclass(frozen=True)
class CalibrationEvidence:
    """Referenced certificate/method and declared origin, not authentication."""

    certificate_sha256: str
    method_id: str
    source_kind: SourceKind

    def __post_init__(self) -> None:
        sha256_reference(self.certificate_sha256, "certificate_sha256")
        identity_text(self.method_id, "method_id")
        instance(self.source_kind, SourceKind, "source_kind")


@dataclass(frozen=True)
class CalibrationRecord:
    """Immutable declared calibration model and the conditions of its use."""

    calibration_id: str
    transform: AffineCalibration
    domain: CalibrationDomain
    evidence: CalibrationEvidence

    def __post_init__(self) -> None:
        identity_text(self.calibration_id, "calibration_id")
        instance(self.transform, AffineCalibration, "transform")
        instance(self.domain, CalibrationDomain, "domain")
        instance(self.evidence, CalibrationEvidence, "evidence")

    def validate_raw(self, raw: RawWaveform) -> tuple[float, float]:
        """Check raw/time/unit compatibility and return the represented valid band."""
        instance(raw, RawWaveform, "raw")
        clock, domain = raw.clock, self.domain
        if raw.unit != self.transform.units[0]:
            raise ValueError("raw indication unit must match calibration input unit")
        if clock.clock_id != domain.clock_id:
            raise ValueError("calibration and acquisition clock identities must match")
        first, last = raw.time_extent_s
        if first < domain.time_interval_s[0] or last > domain.time_interval_s[1]:
            raise ValueError("sample time lies outside calibration validity")
        lower, upper = domain.indication_range
        if np.any(raw.samples < lower) or np.any(raw.samples > upper):
            raise ValueError("raw indication lies outside calibration range")
        low, high = domain.frequency_band_hz
        high = min(high, clock.sample_rate_hz / 2)
        if high <= low:
            raise ValueError(
                "calibration has no positive-width represented frequency band"
            )
        return low, high


__all__ = ()
