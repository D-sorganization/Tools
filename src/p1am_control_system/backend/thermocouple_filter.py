"""Deglitch / hold-last-good / fail-safe filter for thermocouple readings.

The P1-04THM is configured for *low-side burnout*: when a thermocouple input
goes open-circuit (a loose terminal, an intermittent junction, a degrading
element at high temperature) the module drives that channel's reading to the
bottom of scale — 0 C. A control loop that believes a spurious 0 C ("cold")
while the crucible is really >1000 C will command MORE heat: the classic
runaway. Feeding a raw burnout-zero straight into the on/off law is therefore a
safety hazard, and the on-module value passes through as a perfectly finite 0.0
(so the controller's non-finite ``TC_FAULT`` check never sees it).

This filter sits between the raw tag->deg C conversion and the controller. It:

  * accepts plausible readings unchanged;
  * rejects an implausible drop to ~0 (or an impossibly large step down) from a
    hot last-good value, HOLDING the last-good instead so the control law never
    acts on a glitch;
  * if the fault PERSISTS past a timeout, declares a hard fault so the caller can
    trip the heater — holding a stale value forever would let it heat blind.

Pure and clock-injected (``now`` is passed in) so it is fully unit-testable.
It never fabricates a *lower* value than reality, so it cannot mask a real
over-temperature; the worst it does on a genuine fault is hold the last hot
reading for up to ``hold_timeout_s`` before tripping (fail-safe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeIs

__all__ = [
    "FilterSample",
    "ThermocoupleDeglitchFilter",
]

# Defaults tuned for a crucible heater (large thermal mass, 0-1400 C range).
_ZERO_FLOOR_C = 5.0  # a reading at/below this is a candidate burnout "0"
_MIN_DROP_C = 30.0  # ...only if it fell at least this far from last-good
_MAX_STEP_DOWN_C = 250.0  # any drop this large in one scan is non-physical
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
        min_drop_c: float = _MIN_DROP_C,
        max_step_down_c: float = _MAX_STEP_DOWN_C,
        hold_timeout_s: float = _HOLD_TIMEOUT_S,
    ) -> None:
        """Configure the filter thresholds.

        Args:
            zero_floor_c: readings at/below this are candidate burnout zeros.
            min_drop_c: minimum fall from last-good for a near-zero to count as a
                glitch (so genuine near-ambient operation is never rejected).
            max_step_down_c: any single-scan drop this large is non-physical for a
                crucible and is rejected even if the value is not near zero.
            hold_timeout_s: how long to hold last-good through a continuous fault
                before declaring a hard fault (trip). Must be > 0.

        Raises:
            ValueError: if any threshold is negative or hold_timeout_s <= 0.
        """
        if zero_floor_c < 0 or min_drop_c < 0 or max_step_down_c < 0:
            raise ValueError("filter thresholds must be non-negative")
        if hold_timeout_s <= 0:
            raise ValueError("hold_timeout_s must be positive")
        self._zero_floor_c = zero_floor_c
        self._min_drop_c = min_drop_c
        self._max_step_down_c = max_step_down_c
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

        drop = self._last_good_c - raw
        burnout_zero = raw <= self._zero_floor_c and drop >= self._min_drop_c
        impossible_step = drop >= self._max_step_down_c
        if burnout_zero or impossible_step:
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
