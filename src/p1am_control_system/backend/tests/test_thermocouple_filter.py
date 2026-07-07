"""Unit tests for the thermocouple deglitch / hold / fail-safe filter."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from thermocouple_filter import (  # noqa: E402
    ThermocoupleDeglitchFilter,
)


def _f() -> ThermocoupleDeglitchFilter:
    return ThermocoupleDeglitchFilter(hold_timeout_s=15.0)


class TestAcceptsPlausibleReadings:
    def test_bootstraps_on_first_reading(self) -> None:
        f = _f()
        s = f.update(1000.0, now=0.0)
        assert s.value_c == 1000.0
        assert s.holding is False
        assert s.fault is False
        assert f.last_good_c == 1000.0

    def test_passes_normal_ramp_unchanged(self) -> None:
        f = _f()
        f.update(1000.0, now=0.0)
        for i, t in enumerate([1005.0, 1010.0, 1008.0, 1012.0], start=1):
            s = f.update(t, now=float(i))
            assert s.value_c == t
            assert s.holding is False

    def test_first_reading_may_legitimately_be_near_zero(self) -> None:
        # Cold start: no last-good yet, so a ~0 reading is trusted, not rejected.
        f = _f()
        s = f.update(1.0, now=0.0)
        assert s.value_c == 1.0
        assert s.holding is False


class TestRejectsBurnoutZeros:
    def test_drop_to_zero_from_hot_is_held(self) -> None:
        f = _f()
        f.update(1150.0, now=0.0)
        s = f.update(0.0, now=1.0)  # module burnout downscale
        assert s.value_c == 1150.0  # held, not 0
        assert s.holding is True
        assert s.fault is False

    def test_near_ambient_dip_not_rejected(self) -> None:
        # last-good 25 C, reads 2 C: only a 23 C drop — below min_drop, so it is a
        # plausible cool reading, not a burnout. Must be accepted.
        f = _f()
        f.update(25.0, now=0.0)
        s = f.update(2.0, now=1.0)
        assert s.value_c == 2.0
        assert s.holding is False

    def test_impossible_large_step_down_is_held(self) -> None:
        # A 400 C crash in one scan (not to zero) is non-physical -> hold.
        f = _f()
        f.update(1150.0, now=0.0)
        s = f.update(700.0, now=1.0)
        assert s.value_c == 1150.0
        assert s.holding is True

    def test_non_finite_is_held(self) -> None:
        f = _f()
        f.update(1150.0, now=0.0)
        for bad in (math.nan, math.inf, None):
            s = f.update(bad, now=2.0)
            assert s.value_c == 1150.0
            assert s.holding is True


class TestFailSafeTimeout:
    def test_persistent_fault_trips_after_timeout(self) -> None:
        f = ThermocoupleDeglitchFilter(hold_timeout_s=15.0)
        f.update(1150.0, now=0.0)
        # Continuous burnout zeros. Just under the timeout -> hold, no fault.
        assert f.update(0.0, now=1.0).fault is False
        assert f.update(0.0, now=15.9).fault is False
        # At/after the timeout since the fault began -> hard fault (trip).
        s = f.update(0.0, now=16.0)
        assert s.fault is True
        assert s.holding is True
        assert s.value_c == 1150.0  # still reports last-good for context

    def test_recovery_clears_hold_and_resets_timer(self) -> None:
        f = ThermocoupleDeglitchFilter(hold_timeout_s=15.0)
        f.update(1150.0, now=0.0)
        f.update(0.0, now=1.0)  # start holding
        f.update(0.0, now=5.0)  # still holding
        good = f.update(1160.0, now=6.0)  # recovers
        assert good.value_c == 1160.0
        assert good.holding is False
        assert good.fault is False
        # A fresh glitch must start a NEW timeout window, not resume the old one.
        assert f.update(0.0, now=7.0).fault is False


class TestConstruction:
    def test_rejects_bad_thresholds(self) -> None:
        with pytest.raises(ValueError):
            ThermocoupleDeglitchFilter(hold_timeout_s=0.0)
        with pytest.raises(ValueError):
            ThermocoupleDeglitchFilter(zero_floor_c=-1.0)

    def test_reset_forgets_history(self) -> None:
        f = _f()
        f.update(1150.0, now=0.0)
        f.reset()
        assert f.last_good_c is None
        # After reset the next reading bootstraps again (even a 0).
        assert f.update(0.0, now=1.0).holding is False
