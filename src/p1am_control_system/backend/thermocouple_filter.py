"""Deglitch / hold-last-good / fail-safe filter for thermocouple readings.

When a thermocouple input goes open-circuit (a loose terminal, an intermittent
junction, a degrading element at high temperature) the P1-04THM drives that
channel to a *burnout rail*. The direction is operator-selectable: **low-side**
burnout rails the reading to ~0 C, **high-side** rails it to ~full scale. Either
way the on-module value passes through as a perfectly finite number (so the
controller's non-finite ``TC_FAULT`` check never sees it), and either way a
control loop that believes the spurious rail can misbehave — a false "cold"
commands MORE heat (runaway), a false "hot" chatters the heater off.

This filter sits between the raw tag->deg C conversion and the controller. It:

  * accepts plausible readings unchanged;
  * rejects an implausible jump toward EITHER rail (~0 or ~full scale), or any
    non-physical single-scan DROP that isn't quite to 0 (a partial low-side
    burnout), HOLDING the last-good instead so the control law never acts on
    a glitch — whichever burnout direction is configured. A large single-scan
    RISE to a mid-scale value is deliberately NOT rejected on magnitude alone
    (issue #3977): it isn't a burnout signature, and could be a legitimate
    fast change or a channel switch to a different probe;
  * if the fault PERSISTS past a timeout, declares a hard fault so the caller can
    trip the heater — holding a stale value forever would let it heat blind.

Pure and clock-injected (``now`` is passed in) so it is fully unit-testable.
On a genuine fault it holds the last *good* reading (not a rail) for up to
``hold_timeout_s`` before tripping, so it can neither mask a real
over-temperature nor be fooled into commanding heat by a burnout rail.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import hardware

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

__all__ = [
    "DEFAULT_MAX_STEP_C",
    "FilterSample",
    "ThermocoupleDeglitchFilter",
]

# Defaults tuned for a crucible heater (large thermal mass, 0-1400 C range).
_ZERO_FLOOR_C = 5.0  # a reading at/below this is a candidate LOW-side burnout "0"
_MIN_JUMP_C = 30.0  # ...only if it jumped at least this far from last-good
# Public so callers can scale it to a channel whose range differs from the
# firmware default (see TemperatureService._build_filter).
# Single-scan DROP this large is non-physical. Rises are NOT gated by this
# threshold -- see update()'s docstring (issue #3977).
DEFAULT_MAX_STEP_C = 250.0
_MAX_STEP_C = DEFAULT_MAX_STEP_C
# Range top; a reading near here is a HIGH-side burnout rail. Sourced from the
# firmware contract rather than re-declared, so it cannot drift (issue #3998).
_FULL_SCALE_C = hardware.THERMOCOUPLE_FULL_SCALE_C
_RAIL_MARGIN_C = 20.0  # how close to full scale counts as the high rail
_HOLD_TIMEOUT_S = 15.0  # hold through glitches this long, then trip (fail-safe)


@dataclass(frozen=True)
class FilterSample:
    """Result of one filter update.

    Attributes:
        value_c: the temperature the caller should USE for control/display, or
            None before any good reading has ever been seen.
        holding: True when this scan's raw reading was rejected and the last-good
            value is being substituted.
        fault: True when the reading has been bad continuously past the hold
            timeout — the caller should trip (fail-safe), not keep holding.
    """

    value_c: float | None
    holding: bool
    fault: bool


class ThermocoupleDeglitchFilter:
    """Stateful deglitch + hold-last-good + fail-safe filter for one channel."""

    def __init__(
        self,
        *,
        zero_floor_c: float = _ZERO_FLOOR_C,
        min_jump_c: float = _MIN_JUMP_C,
        max_step_c: float = _MAX_STEP_C,
        full_scale_c: float = _FULL_SCALE_C,
        rail_margin_c: float = _RAIL_MARGIN_C,
        hold_timeout_s: float = _HOLD_TIMEOUT_S,
    ) -> None:
        """Configure the filter thresholds.

        The filter is burnout-direction agnostic: the P1-04THM can be set for
        LOW-side burnout (an open reads ~0 C) or HIGH-side burnout (an open reads
        ~full scale), and this rejects an implausible jump toward *either* rail.

        Args:
            zero_floor_c: readings at/below this are candidate low-side burnout "0"s.
            min_jump_c: minimum jump from last-good for a near-rail reading to count
                as a glitch (so genuine near-ambient or near-full-scale operation is
                never rejected).
            max_step_c: any single-scan DROP this large is non-physical for a
                crucible and is rejected even away from a rail. A rise this
                large is NOT gated by this threshold (issue #3977) -- it could
                be a legitimate fast change or a channel switch to a different
                probe, so only the burnout-rail checks apply to rises.
            full_scale_c: the channel range top; a reading within ``rail_margin_c``
                of it is a candidate high-side burnout rail.
            rail_margin_c: how close to full scale counts as the high rail.
            hold_timeout_s: how long to hold last-good through a continuous fault
                before declaring a hard fault (trip). Must be > 0.

        Raises:
            ValueError: if any threshold is negative, full_scale_c <= 0, or
                hold_timeout_s <= 0.
        """
        if min(zero_floor_c, min_jump_c, max_step_c, rail_margin_c) < 0:
            raise ValueError("filter thresholds must be non-negative")
        if full_scale_c <= 0:
            raise ValueError("full_scale_c must be positive")
        if hold_timeout_s <= 0:
            raise ValueError("hold_timeout_s must be positive")
        self._zero_floor_c = zero_floor_c
        self._min_jump_c = min_jump_c
        self._max_step_c = max_step_c
        self._full_scale_c = full_scale_c
        self._rail_margin_c = rail_margin_c
        self._hold_timeout_s = hold_timeout_s
        self._last_good_c: float | None = None
        self._hold_since: float | None = None

    @property
    def last_good_c(self) -> float | None:
        """The most recent accepted reading (None until the first one)."""
        return self._last_good_c

    def reset(self) -> None:
        """Forget all history (e.g. when the channel's source changes)."""
        self._last_good_c = None
        self._hold_since = None

    def update(self, raw_c: float | None, now: float) -> FilterSample:
        """Feed one raw reading and return the value the caller should use.

        Args:
            raw_c: the raw thermocouple reading in deg C, or None when the scan
                produced no data (treated the same as a non-finite reading).
            now: a monotonic timestamp in seconds (injected so this is testable).

        Precondition: ``now`` is a finite float.
        Postcondition: on an accepted reading the internal last-good advances and
        any hold is cleared; on a rejected reading the last-good is held and the
        hold timer runs.
        """
        if not self._is_finite_number(raw_c):
            return self._reject(now)

        raw = float(raw_c)  # narrowed to a finite number by the guard above
        if self._last_good_c is None:
            # Bootstrap: nothing to compare against yet, so trust the first real
            # reading (the controller is IDLE at startup, so this cannot energize).
            return self._accept(raw)

        # A burnout rails the reading to a limit (0 low-side, full scale high-side),
        # so reject an instantaneous jump TOWARD either rail. Also reject any huge
        # single-scan *drop* that isn't quite to 0 (a partial low-side burnout).
        # A large jump to a mid-scale value is left alone: it isn't a burnout
        # signature and could be a legitimate fast change or a channel switch.
        drop = self._last_good_c - raw  # +ve = fell, -ve = rose
        rise = -drop
        burnout_low = raw <= self._zero_floor_c and drop >= self._min_jump_c
        burnout_high = (
            raw >= self._full_scale_c - self._rail_margin_c and rise >= self._min_jump_c
        )
        impossible_drop = drop >= self._max_step_c
        if burnout_low or burnout_high or impossible_drop:
            return self._reject(now)

        return self._accept(raw)

    @staticmethod
    def _is_finite_number(value: float | None) -> TypeIs[float]:
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def _accept(self, raw: float) -> FilterSample:
        self._last_good_c = raw
        self._hold_since = None
        return FilterSample(value_c=raw, holding=False, fault=False)

    def _reject(self, now: float) -> FilterSample:
        if self._hold_since is None:
            self._hold_since = now
        held_for = now - self._hold_since
        fault = held_for >= self._hold_timeout_s
        return FilterSample(value_c=self._last_good_c, holding=True, fault=fault)
