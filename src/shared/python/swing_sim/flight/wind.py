"""Pure, reproducible three-dimensional wind contracts and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

Vector3: TypeAlias = tuple[float, float, float]
WIND_SCHEMA_VERSION = "wind-scenario/v1"
_HARMONIC_COUNT = 6
_TURBULENCE_NORMALIZER = math.sqrt(_HARMONIC_COUNT)


def _vector(value: object, name: str) -> Vector3:
    candidate = np.asarray(value, dtype=float)
    if candidate.shape != (3,) or not bool(np.all(np.isfinite(candidate))):
        raise ValueError(f"{name} must contain three finite components")
    return float(candidate[0]), float(candidate[1]), float(candidate[2])


@dataclass(frozen=True)
class WindGust:
    """One declared gust with a smooth squared-sine envelope."""

    start_time_s: float
    duration_s: float
    peak_velocity_mps: Vector3

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_time_s) or self.start_time_s < 0.0:
            raise ValueError("start_time_s must be finite and nonnegative")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        object.__setattr__(
            self,
            "peak_velocity_mps",
            _vector(self.peak_velocity_mps, "peak_velocity_mps"),
        )

    def velocity_at(self, time_s: float) -> Vector3:
        """Return the gust contribution at physical trajectory time."""
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        elapsed = time_s - self.start_time_s
        if elapsed < 0.0 or elapsed > self.duration_s:
            return 0.0, 0.0, 0.0
        envelope = math.sin(math.pi * elapsed / self.duration_s) ** 2
        return (
            envelope * self.peak_velocity_mps[0],
            envelope * self.peak_velocity_mps[1],
            envelope * self.peak_velocity_mps[2],
        )


_UINT32_MASK = 0xFFFFFFFF
_UINT32_SCALE = 4294967296.0


def _fmix32(h: int) -> int:
    h &= _UINT32_MASK
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _UINT32_MASK
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _UINT32_MASK
    h ^= h >> 16
    return h


def _noise_hash(seed: int, axis: int, harmonic: int, stream: int) -> int:
    h = ((seed & _UINT32_MASK) + 0x9E3779B9) & _UINT32_MASK
    h = (h ^ ((axis + 1) * 0x1E35A7BD)) & _UINT32_MASK
    h = (h ^ ((harmonic + 1) * 0x85EBCA6B)) & _UINT32_MASK
    h = (h ^ ((stream + 1) * 0xC2B2AE35)) & _UINT32_MASK
    return _fmix32(h)


def _unit_noise(seed: int, axis: int, harmonic: int, time_s: float) -> float:
    phase = (_noise_hash(seed, axis, harmonic, 0) / _UINT32_SCALE) * 2.0 * math.pi
    coefficient = (_noise_hash(seed, axis, harmonic, 1) / _UINT32_SCALE) * 2.0 - 1.0
    frequency_hz = 0.2 + 0.27 * harmonic
    return coefficient * math.sin(2.0 * math.pi * frequency_hz * time_s + phase)


@dataclass(frozen=True)
class WindScenario:
    """Versioned wind-to velocity field in the flight frame.

    The frame is x forward, y left, z up. ``base_velocity_mps`` therefore
    describes where the air is moving *to*, not the bearing it comes from.
    """

    base_velocity_mps: Vector3 = (0.0, 0.0, 0.0)
    shear_fraction_per_10m: float = 0.0
    gusts: tuple[WindGust, ...] = ()
    turbulence_intensity_mps: float = 0.0
    seed: int = 0
    provenance: str = "user_declared"
    schema_version: str = WIND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WIND_SCHEMA_VERSION:
            raise ValueError(f"unsupported wind schema: {self.schema_version}")
        object.__setattr__(
            self,
            "base_velocity_mps",
            _vector(self.base_velocity_mps, "base_velocity_mps"),
        )
        object.__setattr__(self, "gusts", tuple(self.gusts))
        for name in ("shear_fraction_per_10m", "turbulence_intensity_mps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not self.provenance.strip():
            raise ValueError("provenance must be nonempty")

    @classmethod
    def from_meteorological(
        cls,
        speed_mps: float,
        from_bearing_deg: float,
        vertical_mps: float = 0.0,
    ) -> WindScenario:
        """Construct from a clockwise bearing the horizontal wind comes from."""
        values = (speed_mps, from_bearing_deg, vertical_mps)
        if not all(math.isfinite(value) for value in values) or speed_mps < 0.0:
            raise ValueError("wind speed/bearing/vertical values must be finite")
        bearing = math.radians(from_bearing_deg)
        velocity = (
            -speed_mps * math.cos(bearing),
            speed_mps * math.sin(bearing),
            vertical_mps,
        )
        return cls(
            base_velocity_mps=velocity,
            provenance="user_declared_meteorological",
        )

    def velocity_at(self, time_s: float, position_m: object) -> Vector3:
        """Evaluate wind-to velocity as a pure function of time and position."""
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        position = _vector(position_m, "position_m")
        altitude_m = max(0.0, position[2])
        shear = 1.0 + self.shear_fraction_per_10m * altitude_m / 10.0
        result = np.asarray(self.base_velocity_mps, dtype=float) * shear
        for gust in self.gusts:
            result += np.asarray(gust.velocity_at(time_s), dtype=float)
        if self.turbulence_intensity_mps > 0.0:
            noise = np.array(
                [
                    sum(
                        _unit_noise(self.seed, axis, harmonic, time_s)
                        for harmonic in range(_HARMONIC_COUNT)
                    )
                    for axis in range(3)
                ]
            )
            result += (self.turbulence_intensity_mps / _TURBULENCE_NORMALIZER) * noise
        return float(result[0]), float(result[1]), float(result[2])

    @property
    def is_steady(self) -> bool:
        """Return whether one constant vector fully describes the scenario."""
        return (
            self.shear_fraction_per_10m == 0.0
            and not self.gusts
            and self.turbulence_intensity_mps == 0.0
        )


__all__ = ["WIND_SCHEMA_VERSION", "WindGust", "WindScenario"]
