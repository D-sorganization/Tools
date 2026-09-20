"""Immutable contracts for measured grip translation/rotation impedance."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from ._validation import require_finite_float, require_identifier

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_KINDS = frozenset({"synthetic", "analytical", "measurement-derived"})


def _digest(value: object, name: str) -> str:
    result: str = require_identifier(value, name)
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return result


class GripAxis(str, Enum):
    """Specific translation or rotation axis in local grip coordinates."""

    TRANSLATION_X = "tx"
    TRANSLATION_Y = "ty"
    TRANSLATION_Z = "tz"
    ROTATION_X = "rx"
    ROTATION_Y = "ry"
    ROTATION_Z = "rz"
    SPATIAL_6DOF = "spatial_6dof"

    @property
    def is_translation(self) -> bool:
        return self in (
            GripAxis.TRANSLATION_X,
            GripAxis.TRANSLATION_Y,
            GripAxis.TRANSLATION_Z,
        )

    @property
    def is_rotation(self) -> bool:
        return self in (
            GripAxis.ROTATION_X,
            GripAxis.ROTATION_Y,
            GripAxis.ROTATION_Z,
        )

    @property
    def spatial_index(self) -> int:
        mapping = {
            GripAxis.TRANSLATION_X: 0,
            GripAxis.TRANSLATION_Y: 1,
            GripAxis.TRANSLATION_Z: 2,
            GripAxis.ROTATION_X: 3,
            GripAxis.ROTATION_Y: 4,
            GripAxis.ROTATION_Z: 5,
        }
        if self in mapping:
            return mapping[self]
        raise ValueError("spatial_6dof has no single spatial index")


@dataclass(frozen=True)
class MeasuredGripSource:
    """Declared coefficient derivation and supporting artifact identities."""

    source_id: str
    kind: str
    artifact_sha256: str
    calibration_sha256: str | None
    method: str
    uncertainty_note: str
    data_license: str

    def __post_init__(self) -> None:
        for name in ("source_id", "kind", "method", "uncertainty_note", "data_license"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )
        if self.kind not in _SOURCE_KINDS:
            raise ValueError(f"unsupported source kind: {self.kind}")
        object.__setattr__(
            self, "artifact_sha256", _digest(self.artifact_sha256, "artifact_sha256")
        )
        if self.calibration_sha256 is not None:
            object.__setattr__(
                self,
                "calibration_sha256",
                _digest(self.calibration_sha256, "calibration_sha256"),
            )
        if self.kind == "measurement-derived" and self.calibration_sha256 is None:
            raise ValueError("measurement-derived source requires calibration_sha256")


@dataclass(frozen=True)
class GripFrequencySample:
    """Measured complex impedance at a single frequency bin with uncertainty."""

    frequency_hz: float
    angular_frequency_rad_s: float
    impedance_real: float
    impedance_imag: float
    magnitude_std: float
    phase_std_rad: float
    is_interpolated: bool = False

    def __post_init__(self) -> None:
        freq = require_finite_float(self.frequency_hz, "frequency_hz", positive=True)
        omega = require_finite_float(
            self.angular_frequency_rad_s, "angular_frequency_rad_s", positive=True
        )
        z_re = require_finite_float(self.impedance_real, "impedance_real")
        z_im = require_finite_float(self.impedance_imag, "impedance_imag")
        mag_std = require_finite_float(self.magnitude_std, "magnitude_std")
        phase_std = require_finite_float(self.phase_std_rad, "phase_std_rad")
        if mag_std < 0.0 or phase_std < 0.0:
            raise ValueError("uncertainty standard deviations must be non-negative")
        if not isinstance(self.is_interpolated, bool):
            raise TypeError("is_interpolated must be a boolean")
        object.__setattr__(self, "frequency_hz", freq)
        object.__setattr__(self, "angular_frequency_rad_s", omega)
        object.__setattr__(self, "impedance_real", z_re)
        object.__setattr__(self, "impedance_imag", z_im)
        object.__setattr__(self, "magnitude_std", mag_std)
        object.__setattr__(self, "phase_std_rad", phase_std)

    @property
    def complex_impedance(self) -> complex:
        return complex(self.impedance_real, self.impedance_imag)

    @property
    def magnitude(self) -> float:
        return abs(self.complex_impedance)

    @property
    def phase_rad(self) -> float:
        return float(math.atan2(self.impedance_imag, self.impedance_real))


@dataclass(frozen=True)
class MeasuredGripDataset:
    """Ordered measured frequency samples for a declared axis and condition."""

    dataset_id: str
    frame_id: str
    axis: GripAxis
    grip_force_n: float
    push_force_n: float
    frequency_band_hz: tuple[float, float]
    sources: tuple[MeasuredGripSource, ...]
    samples: tuple[GripFrequencySample, ...]

    def __post_init__(self) -> None:
        for name in ("dataset_id", "frame_id"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )
        if not isinstance(self.axis, GripAxis):
            raise TypeError("axis must be GripAxis")
        g_force = require_finite_float(self.grip_force_n, "grip_force_n")
        p_force = require_finite_float(self.push_force_n, "push_force_n")
        if g_force < 0.0 or p_force < 0.0:
            raise ValueError("grip and push forces must be non-negative")
        band = tuple(
            require_finite_float(v, f"band[{i}]", positive=True)
            for i, v in enumerate(self.frequency_band_hz)
        )
        if len(band) != 2 or band[0] >= band[1]:
            raise ValueError(
                "frequency_band_hz must be (f_min, f_max) with f_min < f_max"
            )
        object.__setattr__(self, "grip_force_n", g_force)
        object.__setattr__(self, "push_force_n", p_force)
        object.__setattr__(self, "frequency_band_hz", (band[0], band[1]))

        sources = tuple(self.sources)
        if not sources:
            raise ValueError("dataset requires at least one source")
        if any(not isinstance(s, MeasuredGripSource) for s in sources):
            raise TypeError("sources must contain MeasuredGripSource")
        object.__setattr__(self, "sources", sources)

        samples = tuple(self.samples)
        if not samples:
            raise ValueError("dataset requires at least one sample")
        if any(not isinstance(s, GripFrequencySample) for s in samples):
            raise TypeError("samples must contain GripFrequencySample")

        # Verify strict ascending frequency ordering and band inclusion
        last_f = -1.0
        for s in samples:
            if s.frequency_hz <= last_f:
                raise ValueError("sample frequencies must be strictly ascending")
            if s.frequency_hz < band[0] - 1e-9 or s.frequency_hz > band[1] + 1e-9:
                raise ValueError("sample frequency is outside declared band")
            last_f = s.frequency_hz
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True)
class GripPassivityAudit:
    """Passivity verification result at a single frequency."""

    frequency_hz: float
    is_passive: bool
    minimum_real_impedance: float
    dissipated_power_w: float
    passivity_margin: float


@dataclass(frozen=True)
class GripFRFAgreement:
    """Magnitude and phase agreement evaluation for a single sample."""

    angular_frequency_rad_s: float
    measured_magnitude: float
    model_magnitude: float
    relative_magnitude_error: float
    phase_error_rad: float
    uncertainty_coverage: float
    is_within_uncertainty: bool


@dataclass(frozen=True)
class FRFAgreementSummary:
    """Comprehensive summary of model-to-measurement FRF agreement."""

    samples: tuple[GripFRFAgreement, ...]
    max_relative_magnitude_error: float
    max_phase_error_rad: float
    coverage_fraction: float
    passivity_satisfied: bool
    strain_qualified: bool
    agreement_qualified: bool


__all__ = [
    "GripAxis",
    "MeasuredGripSource",
    "GripFrequencySample",
    "MeasuredGripDataset",
    "GripPassivityAudit",
    "GripFRFAgreement",
    "FRFAgreementSummary",
]
