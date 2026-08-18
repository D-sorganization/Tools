"""Deterministic wind-estimate uncertainty sampling contracts.

The versioned schema uses SI speed units and meteorological bearings: bearings
are degrees clockwise from the flight-frame +x target direction and describe
where wind comes *from*. Wind vectors produced from samples use the canonical
flight frame (x forward, y left, z up) and describe where air moves *to*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .wind import WindScenario

WIND_UNCERTAINTY_SCHEMA_VERSION = "wind-uncertainty/v1"
DistributionKind = Literal["fixed", "normal", "uniform"]
_UINT32_MASK = 0xFFFFFFFF
_UINT32_SCALE = float(2**32)
_MIN_UNIFORM = 1.0 / _UINT32_SCALE
_PRECISION_DIGITS = 9


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _rounded(value: float) -> float:
    return round(value, _PRECISION_DIGITS)


def _normalize_bearing(value: float) -> float:
    return _rounded((value + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class ScalarDistribution:
    """One deterministic scalar distribution.

    ``spread`` is the standard deviation for ``normal`` and the symmetric
    half-width for ``uniform``. Optional bounds clamp the sampled value.
    """

    kind: DistributionKind
    center: float
    spread: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("fixed", "normal", "uniform"):
            raise ValueError(f"unsupported distribution kind: {self.kind}")
        _finite(self.center, "center")
        _finite(self.spread, "spread")
        if self.spread < 0.0:
            raise ValueError("spread must be nonnegative")
        if self.kind == "fixed" and self.spread != 0.0:
            raise ValueError("fixed distributions require zero spread")
        for name, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if value is not None:
                _finite(value, name)
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("minimum must not exceed maximum")

    def clamp(self, value: float) -> float:
        """Apply declared bounds and return a parity-quantized sample."""
        if self.minimum is not None:
            value = max(self.minimum, value)
        if self.maximum is not None:
            value = min(self.maximum, value)
        return _rounded(value)


@dataclass(frozen=True)
class WindEstimateError:
    """Correlated player-estimate error in speed and from-bearing."""

    speed_bias_mps: float = 0.0
    speed_std_mps: float = 0.0
    bearing_bias_deg: float = 0.0
    bearing_std_deg: float = 0.0
    correlation: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "speed_bias_mps",
            "speed_std_mps",
            "bearing_bias_deg",
            "bearing_std_deg",
            "correlation",
        ):
            _finite(getattr(self, name), name)
        if self.speed_std_mps < 0.0 or self.bearing_std_deg < 0.0:
            raise ValueError("estimate standard deviations must be nonnegative")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must be in [-1, 1]")


@dataclass(frozen=True)
class WindUncertaintySpec:
    """Versioned true-wind and player-estimate ensemble specification."""

    trials: int
    seed: int
    true_speed_mps: ScalarDistribution
    true_from_bearing_deg: ScalarDistribution
    estimate_error: WindEstimateError = WindEstimateError()
    provenance: str = "user_declared"
    schema_version: str = WIND_UNCERTAINTY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WIND_UNCERTAINTY_SCHEMA_VERSION:
            raise ValueError(f"unsupported uncertainty schema: {self.schema_version}")
        if not isinstance(self.trials, int) or not 1 <= self.trials <= 100_000:
            raise ValueError("trials must be an integer in [1, 100000]")
        if not isinstance(self.seed, int) or not 0 <= self.seed <= _UINT32_MASK:
            raise ValueError("seed must be a uint32 integer")
        if not self.provenance.strip():
            raise ValueError("provenance must be nonempty")
        if self.true_speed_mps.minimum is None:
            raise ValueError("true_speed_mps requires a nonnegative minimum")
        if self.true_speed_mps.minimum < 0.0:
            raise ValueError("true_speed_mps minimum must be nonnegative")


@dataclass(frozen=True)
class WindTrial:
    """One common-random-number true/estimated wind pair."""

    trial_index: int
    true_speed_mps: float
    true_from_bearing_deg: float
    estimated_speed_mps: float
    estimated_from_bearing_deg: float
    speed_error_mps: float
    bearing_error_deg: float

    def __post_init__(self) -> None:
        if not isinstance(self.trial_index, int) or self.trial_index < 0:
            raise ValueError("trial_index must be a nonnegative integer")
        values = (
            self.true_speed_mps,
            self.true_from_bearing_deg,
            self.estimated_speed_mps,
            self.estimated_from_bearing_deg,
            self.speed_error_mps,
            self.bearing_error_deg,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("wind trial values must be finite")
        if self.true_speed_mps < 0.0 or self.estimated_speed_mps < 0.0:
            raise ValueError("wind trial speeds must be nonnegative")
        for bearing in (
            self.true_from_bearing_deg,
            self.estimated_from_bearing_deg,
        ):
            if not -180.0 <= bearing < 180.0:
                raise ValueError("wind trial bearings must be in [-180, 180)")

    def true_scenario(self, provenance: str) -> WindScenario:
        """Return the true meteorological wind as a flight-frame field."""
        scenario = WindScenario.from_meteorological(
            self.true_speed_mps, self.true_from_bearing_deg
        )
        return WindScenario(
            base_velocity_mps=scenario.base_velocity_mps,
            provenance=f"{provenance}/true/trial-{self.trial_index}",
        )

    def estimated_scenario(self, provenance: str) -> WindScenario:
        """Return the player-estimated meteorological wind field."""
        scenario = WindScenario.from_meteorological(
            self.estimated_speed_mps, self.estimated_from_bearing_deg
        )
        return WindScenario(
            base_velocity_mps=scenario.base_velocity_mps,
            provenance=f"{provenance}/estimated/trial-{self.trial_index}",
        )

    def to_schema_dict(self) -> dict[str, int | float]:
        """Serialize with exact snake-case cross-language field names."""
        return {
            "trial_index": self.trial_index,
            "true_speed_mps": self.true_speed_mps,
            "true_from_bearing_deg": self.true_from_bearing_deg,
            "estimated_speed_mps": self.estimated_speed_mps,
            "estimated_from_bearing_deg": self.estimated_from_bearing_deg,
            "speed_error_mps": self.speed_error_mps,
            "bearing_error_deg": self.bearing_error_deg,
        }


class _Mulberry32:
    """Small uint32 PRNG duplicated exactly in the TypeScript twin."""

    def __init__(self, seed: int) -> None:
        self._state = seed & _UINT32_MASK

    def uniform(self) -> float:
        """Return a deterministic sample in [0, 1)."""
        self._state = (self._state + 0x6D2B79F5) & _UINT32_MASK
        value = self._state
        value = _imul(value ^ (value >> 15), value | 1)
        value ^= (value + _imul(value ^ (value >> 7), value | 61)) & _UINT32_MASK
        value &= _UINT32_MASK
        return ((value ^ (value >> 14)) & _UINT32_MASK) / _UINT32_SCALE

    def standard_normal(self) -> float:
        """Return one Box-Muller normal while consuming two uniforms."""
        first = max(_MIN_UNIFORM, self.uniform())
        second = self.uniform()
        return math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)


def _imul(left: int, right: int) -> int:
    return (left * right) & _UINT32_MASK


def _draw(distribution: ScalarDistribution, generator: _Mulberry32) -> float:
    if distribution.kind == "fixed":
        value = distribution.center
    elif distribution.kind == "uniform":
        value = distribution.center + distribution.spread * (
            2.0 * generator.uniform() - 1.0
        )
    else:
        value = distribution.center + distribution.spread * generator.standard_normal()
    return distribution.clamp(value)


def sample_wind_trials(spec: WindUncertaintySpec) -> tuple[WindTrial, ...]:
    """Generate reproducible paired true and player-estimated wind draws."""
    generator = _Mulberry32(spec.seed)
    trials: list[WindTrial] = []
    correlation_scale = math.sqrt(max(0.0, 1.0 - spec.estimate_error.correlation**2))
    for trial_index in range(spec.trials):
        true_speed = _draw(spec.true_speed_mps, generator)
        true_bearing = _normalize_bearing(_draw(spec.true_from_bearing_deg, generator))
        speed_normal = generator.standard_normal()
        independent_normal = generator.standard_normal()
        bearing_normal = (
            spec.estimate_error.correlation * speed_normal
            + correlation_scale * independent_normal
        )
        speed_error = _rounded(
            spec.estimate_error.speed_bias_mps
            + spec.estimate_error.speed_std_mps * speed_normal
        )
        bearing_error = _rounded(
            spec.estimate_error.bearing_bias_deg
            + spec.estimate_error.bearing_std_deg * bearing_normal
        )
        estimated_speed = _rounded(max(0.0, true_speed + speed_error))
        trials.append(
            WindTrial(
                trial_index=trial_index,
                true_speed_mps=true_speed,
                true_from_bearing_deg=true_bearing,
                estimated_speed_mps=estimated_speed,
                estimated_from_bearing_deg=_normalize_bearing(
                    true_bearing + bearing_error
                ),
                speed_error_mps=_rounded(estimated_speed - true_speed),
                bearing_error_deg=bearing_error,
            )
        )
    return tuple(trials)


__all__ = [
    "WIND_UNCERTAINTY_SCHEMA_VERSION",
    "ScalarDistribution",
    "WindEstimateError",
    "WindTrial",
    "WindUncertaintySpec",
    "sample_wind_trials",
]
