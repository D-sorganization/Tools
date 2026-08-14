"""Monotonic-deadline scan scheduling and overrun detection (issue #4009).

A fixed ``sleep(interval)`` after the work makes the real control period
``t_work + interval`` and lets it drift with load. The scheduler instead aims at
a monotonic deadline, counts every missed one, and resynchronises the phase so
lag cannot accumulate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from performance import ScanScheduler  # noqa: E402


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestScanScheduler:
    def test_sleep_absorbs_the_work_time(self) -> None:
        clock = _Clock()
        sched = ScanScheduler(0.1, monotonic=clock)

        clock.advance(0.04)  # the scan took 40 ms
        assert sched.next_sleep_s() == pytest.approx(0.06)
        assert sched.overrun_count == 0

    def test_period_does_not_drift_across_cycles(self) -> None:
        clock = _Clock()
        sched = ScanScheduler(0.1, monotonic=clock)

        starts = []
        for work_s in (0.04, 0.01, 0.07, 0.02):
            starts.append(clock.now)
            clock.advance(work_s)
            clock.advance(sched.next_sleep_s())

        deltas = [b - a for a, b in zip(starts, starts[1:], strict=False)]
        assert deltas == pytest.approx([0.1, 0.1, 0.1])

    def test_overrun_is_counted_measured_and_resynchronised(self) -> None:
        clock = _Clock()
        sched = ScanScheduler(0.1, monotonic=clock)

        clock.advance(0.35)  # a 3 s Modbus timeout blew the deadline
        assert sched.next_sleep_s() == 0.0
        assert sched.overrun_count == 1
        assert sched.last_overrun_s == pytest.approx(0.25)

        # Phase is resynchronised: the next cycle is a full period away, not
        # four back-to-back catch-up scans.
        clock.advance(0.02)
        assert sched.next_sleep_s() == pytest.approx(0.08)
        assert sched.overrun_count == 1

    def test_resync_rebases_the_deadline_after_a_failure_backoff(self) -> None:
        clock = _Clock()
        sched = ScanScheduler(0.1, monotonic=clock)

        sched.resync(0.8)  # the loop is backing off after a failed scan
        clock.advance(0.6)
        assert sched.next_sleep_s() == pytest.approx(0.2)
        # The backoff is not charged as an overrun — it was deliberate.
        assert sched.overrun_count == 0

    def test_set_period_takes_effect_on_the_next_cycle(self) -> None:
        clock = _Clock()
        sched = ScanScheduler(0.1, monotonic=clock)

        sched.set_period_s(0.5)
        assert sched.period_s == 0.5
        clock.advance(0.2)
        assert sched.next_sleep_s() == pytest.approx(0.3)

    def test_validates_the_period(self) -> None:
        with pytest.raises(ValueError):
            ScanScheduler(0.0)
        with pytest.raises(ValueError):
            ScanScheduler(float("inf"))
        with pytest.raises(TypeError):
            ScanScheduler("fast")
        with pytest.raises(TypeError):
            ScanScheduler(True)

    def test_resync_validates_the_delay(self) -> None:
        sched = ScanScheduler(0.1)
        with pytest.raises(ValueError):
            sched.resync(-1.0)
        with pytest.raises(TypeError):
            sched.resync(None)
