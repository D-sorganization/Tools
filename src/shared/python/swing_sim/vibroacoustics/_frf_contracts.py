"""Immutable, finite numerical H1 contracts; no calibration qualification."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Complex, Real

import numpy as np


def nonnegative_real(value: object, name: str) -> float:
    """Accept finite nonnegative real scalars, never Boolean/string coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-Boolean number")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class H1Settings:
    """Explicit segment length and strict PSD support floors in each unit²/Hz.

    Zero selects algebraic positivity only. Floors do not authenticate a sensor
    noise floor, usable bandwidth or statistical confidence.
    """

    segment_length: int
    minimum_input_psd: float
    minimum_response_psd: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.segment_length, bool)
            or not isinstance(self.segment_length, int)
            or self.segment_length < 3
        ):
            raise ValueError("segment length must be an integer >= 3")
        for name in ("minimum_input_psd", "minimum_response_psd"):
            object.__setattr__(
                self, name, nonnegative_real(getattr(self, name), "PSD floor")
            )


@dataclass(frozen=True)
class H1Bin:
    """One bin: declared-unit PSDs, complex response/input and optional coherence.

    None means unsupported/undefined, not zero. A zero complex transfer has no
    defined phase. Coherence must be in [0,1] with positive input/response power.
    """

    input_psd: float
    response_psd: float
    h1: complex | None
    coherence: float | None

    def __post_init__(self) -> None:
        for name in ("input_psd", "response_psd"):
            object.__setattr__(self, name, nonnegative_real(getattr(self, name), name))
        if self.h1 is not None:
            if isinstance(self.h1, (bool, np.bool_)) or not isinstance(
                self.h1, Complex
            ):
                raise TypeError("H1 must be a finite complex scalar or None")
            if not np.isfinite(self.h1) or self.input_psd <= 0:
                raise ValueError("H1 requires finite transfer and positive input PSD")
            object.__setattr__(self, "h1", complex(self.h1))
        if self.coherence is not None:
            value = nonnegative_real(self.coherence, "coherence")
            if value > 1 or self.h1 is None or self.response_psd <= 0:
                raise ValueError(
                    "coherence requires supported H1, positive response PSD and [0,1]"
                )
            object.__setattr__(self, "coherence", value)


@dataclass(frozen=True)
class H1Estimate:
    """Immutable frequency/bin tuples and complete-frame count.

    Units are (input, response) declarations. Segment count is not independent
    degrees of freedom; this object supplies no uncertainty or measured status.
    """

    frequencies_hz: tuple[float, ...]
    bins: tuple[H1Bin, ...]
    segment_count: int
    units: tuple[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.frequencies_hz, tuple) or not isinstance(
            self.bins, tuple
        ):
            raise TypeError("frequency and bin axes must be immutable tuples")
        if not self.bins or len(self.frequencies_hz) != len(self.bins):
            raise ValueError("frequency and bin axes must have equal nonzero lengths")
        values = tuple(
            nonnegative_real(value, "frequency") for value in self.frequencies_hz
        )
        if values[0] != 0 or any(
            right <= left for left, right in zip(values, values[1:], strict=False)
        ):
            raise ValueError("frequencies must start at DC and strictly increase")
        if not all(isinstance(item, H1Bin) for item in self.bins):
            raise TypeError("bins must be H1Bin instances")
        if (
            isinstance(self.segment_count, bool)
            or not isinstance(self.segment_count, int)
            or self.segment_count < 2
        ):
            raise ValueError("at least two complete spectral segments are required")
        if (
            not isinstance(self.units, tuple)
            or len(self.units) != 2
            or not all(isinstance(value, str) and value.strip() for value in self.units)
        ):
            raise ValueError("units must be two nonempty string declarations")
        object.__setattr__(self, "frequencies_hz", values)


__all__ = ()
