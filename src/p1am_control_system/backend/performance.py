"""Scan cadence for the PLC control loop.

Two separate concerns live here, deliberately decoupled (issue #4008):

* :class:`ScanScheduler` owns THE control period — the rate at which the PLC is
  scanned, alarms are evaluated, the heater relay is decided and the E-stop is
  re-asserted. It is fixed by ``settings.poll_interval_s`` and scheduled against
  a monotonic deadline so it neither drifts with load nor accumulates lag.
* :class:`PerformanceController` owns how often the live frame is pushed to the
  HMI. The browser lowers this when its tab is hidden; that is a rendering
  concession and it may only decimate the *broadcast*, never the control period.

Both are pure and dependency-light (clocks are injected) so the cadence contract
is unit-testable without sleeping in an infinite loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from performance_models import PerformanceConfig, PerformanceMode

__all__ = [
    "PerformanceConfig",
    "PerformanceController",
    "PerformanceMode",
    "ScanScheduler",
]


def _validate_interval(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Validate a finite, positive (or non-negative) seconds value.

    Raises:
        TypeError: if ``value`` is not a non-bool int/float.
        ValueError: if ``value`` is non-finite or out of range.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {value}")
    if allow_zero:
        if v < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    elif v <= 0.0:
        raise ValueError(f"{name} must be a finite value > 0, got {value}")
    return v


class ScanScheduler:
    """Monotonic-deadline scheduler with overrun detection (issue #4009).

    ``sleep(period)`` *after* the work makes the true period ``t_work + period``
    and lets it drift with CPU load. This instead advances a deadline by exactly
    one period per cycle and sleeps only the remainder. When the deadline has
    already passed the cycle is counted as an overrun and the phase is
    resynchronised, so a transient stall (e.g. a 3 s Modbus timeout) does not
    produce a burst of back-to-back catch-up scans.

    DbC: the period must be finite and > 0; a clock is injected for tests.
    """

    def __init__(
        self,
        period_s: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self._period_s = _validate_interval(period_s, "period_s")
        self._clock = monotonic
        self._next_deadline = self._clock() + self._period_s
        self._overrun_count = 0
        self._last_overrun_s = 0.0

    @property
    def period_s(self) -> float:
        """The scheduled control period in seconds."""
        return self._period_s

    @property
    def overrun_count(self) -> int:
        """Cycles that missed their deadline since the last counter reset."""
        return self._overrun_count

    @property
    def last_overrun_s(self) -> float:
        """How late the most recent overrun was, in seconds (0.0 if none)."""
        return self._last_overrun_s

    def set_period_s(self, value: float) -> None:
        """Change the control period and re-phase on it from now.

        Re-phasing avoids a spurious overrun (or a long idle) on the one cycle
        that straddles the change.

        Raises:
            TypeError: if ``value`` is not numeric.
            ValueError: if ``value`` is not finite and > 0.
        """
        self._period_s = _validate_interval(value, "period_s")
        self._next_deadline = self._clock() + self._period_s

    def next_sleep_s(self) -> float:
        """Seconds to sleep to land on the next deadline.

        Returns ``0.0`` when the deadline has already passed; that cycle is
        recorded as an overrun and the phase is rebased on *now* so lag cannot
        accumulate across cycles.
        """
        now = self._clock()
        remaining = self._next_deadline - now
        if remaining < 0.0:
            self._overrun_count += 1
            self._last_overrun_s = -remaining
            self._next_deadline = now + self._period_s
            return 0.0
        self._next_deadline += self._period_s
        return remaining

    def resync(self, delay_s: float) -> None:
        """Rebase the deadline ``delay_s`` from now (used after a failure backoff).

        Raises:
            TypeError: if ``delay_s`` is not numeric.
            ValueError: if ``delay_s`` is negative or non-finite.
        """
        delay = _validate_interval(delay_s, "delay_s", allow_zero=True)
        self._next_deadline = self._clock() + delay


class PerformanceController:
    """Holds the active mode and resolves it to a *broadcast* cadence.

    DbC: all intervals must be finite and > 0; ``set_mode`` rejects non-enum
    input. Deliberately exposes no ``poll_interval_s`` — the control period is
    owned by :class:`ScanScheduler` and must never be readable off a UI-driven
    mode (that was the #4008 defect).
    """

    def __init__(
        self,
        performance_interval_s: float,
        lightweight_interval_s: float,
        mode: PerformanceMode = PerformanceMode.PERFORMANCE,
        *,
        scan_interval_s: float | None = None,
    ) -> None:
        self._performance_s = _validate_interval(
            performance_interval_s, "performance_interval_s"
        )
        self._lightweight_s = _validate_interval(
            lightweight_interval_s, "lightweight_interval_s"
        )
        self._scan_s = (
            self._performance_s
            if scan_interval_s is None
            else _validate_interval(scan_interval_s, "scan_interval_s")
        )
        if not isinstance(mode, PerformanceMode):
            raise TypeError(
                f"mode must be a PerformanceMode, got {type(mode).__name__}"
            )
        self._mode = mode

    @property
    def mode(self) -> PerformanceMode:
        return self._mode

    @property
    def scan_interval_s(self) -> float:
        """The fixed control period — reported, never selected, by the mode."""
        return self._scan_s

    @property
    def broadcast_interval_s(self) -> float:
        """Target seconds between WebSocket frames for the active mode."""
        if self._mode == PerformanceMode.LIGHTWEIGHT:
            return self._lightweight_s
        return self._performance_s

    @property
    def broadcast_every_n(self) -> int:
        """Scans per broadcast: how the mode decimates the live stream.

        Always >= 1, so even the slowest mode still streams — and the scan loop
        keeps running at full rate underneath it.
        """
        return max(1, round(self.broadcast_interval_s / self._scan_s))

    def set_mode(self, mode: PerformanceMode) -> None:
        """Switch the active mode.

        Raises:
            TypeError: if mode is not a PerformanceMode.
        """
        if not isinstance(mode, PerformanceMode):
            raise TypeError(
                f"mode must be a PerformanceMode, got {type(mode).__name__}"
            )
        self._mode = mode

    def config(
        self,
        *,
        scan_overruns: int = 0,
        historian_write_failures: int = 0,
    ) -> PerformanceConfig:
        """Report the mode, both cadences and the loop-health counters.

        Raises:
            TypeError: if a counter is not an int.
            ValueError: if a counter is negative.
        """
        for name, value in (
            ("scan_overruns", scan_overruns),
            ("historian_write_failures", historian_write_failures),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        return PerformanceConfig(
            mode=self._mode,
            poll_interval_s=self.broadcast_interval_s,
            broadcast_interval_s=self.broadcast_interval_s,
            scan_interval_s=self._scan_s,
            broadcast_every_n=self.broadcast_every_n,
            scan_overruns=scan_overruns,
            historian_write_failures=historian_write_failures,
        )
