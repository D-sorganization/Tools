"""Vendor-neutral camera identity and capability records."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_finite, require_text
from .enums import ShutterKind, SupportLevel


@dataclass(frozen=True, slots=True)
class NumericRange:
    """Inclusive finite range with an explicit unit."""

    minimum: float
    maximum: float
    unit: str

    def __post_init__(self) -> None:
        minimum = require_finite(self.minimum, "minimum")
        maximum = require_finite(self.maximum, "maximum")
        if minimum > maximum:
            raise ValueError("minimum must not exceed maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "unit", require_text(self.unit, "unit"))

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies inside this inclusive range."""
        normalized = require_finite(value, "value")
        return self.minimum <= normalized <= self.maximum


@dataclass(frozen=True, slots=True)
class FeatureSupport:
    """A capability support result with an actionable degraded reason."""

    level: SupportLevel
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.level, SupportLevel):
            raise TypeError("level must be a SupportLevel")
        reason = None if self.reason is None else require_text(self.reason, "reason")
        if self.level is not SupportLevel.SUPPORTED and reason is None:
            raise ValueError("reason is required for degraded or unsupported support")
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class CameraIdentity:
    """Stable provider identity separate from transient connection address."""

    provider_id: str
    device_id: str
    transport: str
    vendor: str | None = None
    model: str | None = None
    serial_number: str | None = None
    firmware_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", require_text(self.provider_id, "provider_id")
        )
        object.__setattr__(self, "device_id", require_text(self.device_id, "device_id"))
        object.__setattr__(self, "transport", require_text(self.transport, "transport"))
        for name in ("vendor", "model", "serial_number", "firmware_version"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_text(value, name))

    @property
    def stable_key(self) -> str:
        """Return the provider-scoped stable device key."""
        return f"{self.provider_id}:{self.device_id}"


@dataclass(frozen=True, slots=True)
class CameraCapabilities:
    """Effective camera modes and negotiated timing/control capabilities."""

    resolutions_px: tuple[tuple[int, int], ...]
    frame_rates_hz: tuple[float, ...]
    pixel_formats: tuple[str, ...]
    shutter: ShutterKind
    hardware_trigger: FeatureSupport
    device_timestamps: FeatureSupport
    exposure_us: NumericRange | None = None

    def __post_init__(self) -> None:
        if not self.resolutions_px:
            raise ValueError("resolutions_px must be non-empty")
        resolutions = tuple(
            self._validate_resolution(value) for value in self.resolutions_px
        )
        if len(set(resolutions)) != len(resolutions):
            raise ValueError("resolutions_px must be unique")
        rates = tuple(
            require_finite(value, "frame_rates_hz") for value in self.frame_rates_hz
        )
        if (
            not rates
            or any(value <= 0.0 for value in rates)
            or len(set(rates)) != len(rates)
        ):
            raise ValueError("frame_rates_hz must contain unique positive rates")
        formats = tuple(
            require_text(value, "pixel_formats") for value in self.pixel_formats
        )
        if not formats or len(set(formats)) != len(formats):
            raise ValueError("pixel_formats must contain unique non-empty values")
        if not isinstance(self.shutter, ShutterKind):
            raise TypeError("shutter must be a ShutterKind")
        object.__setattr__(self, "resolutions_px", resolutions)
        object.__setattr__(self, "frame_rates_hz", rates)
        object.__setattr__(self, "pixel_formats", formats)

    @staticmethod
    def _validate_resolution(value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or any(
            isinstance(item, bool) or not isinstance(item, int) for item in value
        ):
            raise TypeError("each resolution must contain two integers")
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("resolution dimensions must be positive")
        return value

    def supports_mode(
        self, resolution_px: tuple[int, int], frame_rate_hz: float, pixel_format: str
    ) -> bool:
        """Return whether each requested mode dimension is advertised."""
        rate = require_finite(frame_rate_hz, "frame_rate_hz")
        normalized_format = require_text(pixel_format, "pixel_format")
        return (
            resolution_px in self.resolutions_px
            and rate in self.frame_rates_hz
            and normalized_format in self.pixel_formats
        )


__all__: list[str] = []
