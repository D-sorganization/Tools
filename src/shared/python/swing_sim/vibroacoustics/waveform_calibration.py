"""Explicit affine calibration with exact identity and shared uncertainty.

These are declared numerical models. Content identity and nominal conversion do
not authenticate metrological traceability, synchronization or acoustic validity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._acquisition_records import (
    AcquisitionChannel,
    AcquisitionClock,
    IndependentSampleUncertainty,
    RawWaveform,
    WaveformProvenance,
)
from ._affine_calibration_records import (
    AffineCalibration,
    CalibrationDomain,
    CalibrationEvidence,
    CalibrationRecord,
    GainOffsetUncertainty,
    UncertaintyPropagation,
)
from ._calibration_identity import calibrated_identity
from ._calibration_primitives import finite_scalar, instance
from ._spectral_frames import finite_output
from ._waveform_contracts import owned_real_samples, real_samples
from .measurement import SourceKind


def _factors(
    raw: RawWaveform, calibration: CalibrationRecord
) -> tuple[np.ndarray, np.ndarray] | None:
    """Factor J_c C_gb J_c^T without losing valid perfect correlations."""
    transform = calibration.transform
    uncertainty = transform.uncertainty
    if uncertainty is None:
        return None
    rho = uncertainty.correlation
    gain_u, offset_u = (
        uncertainty.gain_standard_uncertainty,
        uncertainty.offset_standard_uncertainty,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        first = finite_output(raw.samples * gain_u + rho * offset_u)
        second = np.full(raw.size, np.sqrt((1 - rho) * (1 + rho)) * offset_u)
    return owned_real_samples(first), owned_real_samples(second)


def _indication_scale(
    calibration: CalibrationRecord, propagation: UncertaintyPropagation
) -> float:
    """Exact moments add Var(G)*Cov(X); first order retains only mean(G)^2."""
    transform = calibration.transform
    uncertainty = transform.uncertainty
    if (
        propagation is UncertaintyPropagation.EXACT_INDEPENDENT
        and uncertainty is not None
    ):
        return float(np.hypot(transform.gain, uncertainty.gain_standard_uncertainty))
    return abs(float(transform.gain))


@dataclass(frozen=True)
class CalibratedWaveform:
    """Apply y=g*x+b once and retain calibration/acquisition/uncertainty inputs.

    Unknown uncertainty remains None. The optional raw uncertainty explicitly
    assumes independent indications and independence from calibration parameters.
    Shared calibration uncertainty is preserved across samples. Select first-order
    or exact independent-block second moments explicitly through propagation.
    Neither supplies coverage probability, clock uncertainty or spectral phase
    calibration. Values are nominal; validity remains limited to the retained
    flat-response band, with no implicit filtering or anti-alias qualification.
    """

    raw: RawWaveform
    calibration: CalibrationRecord
    sample_uncertainty: IndependentSampleUncertainty | None = None
    propagation: UncertaintyPropagation = UncertaintyPropagation.FIRST_ORDER
    values: np.ndarray = field(init=False, repr=False)
    valid_frequency_band_hz: tuple[float, float] = field(init=False)
    _shared_factors: tuple[np.ndarray, np.ndarray] | None = field(
        init=False, repr=False
    )
    identity_sha256: str = field(init=False)
    _raw_uncertainty_scale: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        instance(self.raw, RawWaveform, "raw")
        instance(self.calibration, CalibrationRecord, "calibration")
        instance(self.propagation, UncertaintyPropagation, "propagation")
        if self.sample_uncertainty is not None:
            instance(
                self.sample_uncertainty,
                IndependentSampleUncertainty,
                "sample_uncertainty",
            )
            values = self.sample_uncertainty.standard_uncertainty
            if values.size != self.raw.size:
                raise ValueError(
                    "raw uncertainty must have the same length as indications"
                )
        band = self.calibration.validate_raw(self.raw)
        transform = self.calibration.transform
        with np.errstate(over="ignore", invalid="ignore"):
            calibrated = finite_output(
                transform.gain * self.raw.samples + transform.offset
            )
        object.__setattr__(self, "values", owned_real_samples(calibrated))
        object.__setattr__(self, "valid_frequency_band_hz", band)
        object.__setattr__(
            self, "_shared_factors", _factors(self.raw, self.calibration)
        )
        object.__setattr__(
            self,
            "_raw_uncertainty_scale",
            _indication_scale(self.calibration, self.propagation),
        )
        object.__setattr__(
            self, "identity_sha256", calibrated_identity(self, self.values)
        )

    @property
    def unit(self) -> str:
        """Return the declared output unit of the nominal conversion."""
        transform = self.calibration.transform
        return transform.units[1]

    @property
    def source_kind(self) -> SourceKind:
        """Synthetic input or calibration cannot become measured by conversion."""
        evidence = self.calibration.evidence
        if (
            self.raw.source_kind is SourceKind.SYNTHESIZED
            or evidence.source_kind is SourceKind.SYNTHESIZED
        ):
            return SourceKind.SYNTHESIZED
        return SourceKind.MEASURED

    def _index(self, value: int) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < self.raw.size
        ):
            raise ValueError("sample index must be an integer within the recording")
        return value

    def calibration_covariance(self, first: int, second: int) -> float | None:
        """Return only the shared calibration component, in output-unit squared."""
        i, j = self._index(first), self._index(second)
        if self._shared_factors is None:
            return None
        a, b = self._shared_factors
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(a[i] * a[j] + b[i] * b[j])
        return finite_scalar(value, "calibration covariance")

    def covariance(self, first: int, second: int) -> float | None:
        """Return the selected covariance law; missing components remain unknown."""
        value = self.calibration_covariance(first, second)
        if value is None or self.sample_uncertainty is None:
            return None
        if first == second:
            sigma = self.sample_uncertainty.standard_uncertainty
            with np.errstate(over="ignore", invalid="ignore"):
                value += float((self._raw_uncertainty_scale * sigma[first]) ** 2)
        return finite_scalar(value, "combined covariance")

    def marginal_standard_uncertainty(self) -> np.ndarray | None:
        """Return combined standard uncertainties, without dropping common errors."""
        if self._shared_factors is None or self.sample_uncertainty is None:
            return None
        a, b = self._shared_factors
        sigma = self.sample_uncertainty.standard_uncertainty
        with np.errstate(over="ignore", invalid="ignore"):
            values = finite_output(
                np.hypot(np.hypot(self._raw_uncertainty_scale * sigma, a), b)
            )
        return owned_real_samples(values)

    def linear_standard_uncertainty(self, weights: np.ndarray) -> float | None:
        """Uncertainty of a dimensionless weighted sum, retaining common errors."""
        weights = real_samples(weights)
        if weights.size != self.raw.size:
            raise ValueError("weights must have the same length as indications")
        if self._shared_factors is None or self.sample_uncertainty is None:
            return None
        a, b = self._shared_factors
        sigma = self.sample_uncertainty.standard_uncertainty
        with np.errstate(over="ignore", invalid="ignore"):
            individual = finite_output(self._raw_uncertainty_scale * weights * sigma)
            shared = finite_output(np.array([weights @ a, weights @ b]))
            value = float(
                np.hypot(np.hypot.reduce(individual), np.hypot.reduce(shared))
            )
        return finite_scalar(value, "linear standard uncertainty")


__all__ = [
    "AcquisitionChannel",
    "AcquisitionClock",
    "WaveformProvenance",
    "RawWaveform",
    "IndependentSampleUncertainty",
    "GainOffsetUncertainty",
    "AffineCalibration",
    "CalibrationDomain",
    "CalibrationEvidence",
    "CalibrationRecord",
    "CalibratedWaveform",
    "UncertaintyPropagation",
]
