"""Runtime / feedback-driven unit tests for PowerSupplyController.

Split companion: `test_power_supply.py` covers config-level + state-machine
behavior. This file covers what happens once the controller is fed measured
values through `tick()`:
    - Current setpoint clamping (fat-finger protection) and type/value
      validation (NaN / inf / bool / string rejection)
    - Power setpoint derivation from measured voltage
    - HH_POWER and HH_TEMP trip latching across safe subsequent ticks
    - Trip acknowledgement + state transitions
    - tick() output guarantees in IDLE / ARMED / TRIPPED / no-permissive
      states and the percentage-output clamp
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _power_supply_helpers import (
    fresh_armed_controller,
    fresh_idle_controller,
    fresh_running_controller,
)
from power_supply import (
    PowerSupplyConfig,
    PowerSupplyController,
    PowerSupplyMode,
    PowerSupplyState,
)

# --------------------------------------------------------------------------
# Current setpoint — fat-finger protection
# --------------------------------------------------------------------------


class TestCurrentSetpoint:
    def test_in_band_setpoint_applied(self) -> None:
        c = fresh_armed_controller()
        applied = c.set_current_setpoint(25.0)
        assert applied == 25.0
        assert c.state == PowerSupplyState.RUNNING

    def test_setpoint_above_max_is_clamped(self) -> None:
        c = fresh_armed_controller()
        applied = c.set_current_setpoint(9999.0)
        assert applied == c.config.current_setpoint_max_a
        c.tick(0.0, 0.0, 25.0, now=0.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=100.0)
        assert cmd == pytest.approx(
            100.0 * c.config.current_setpoint_max_a / c.config.current_full_scale_a
        )

    def test_setpoint_below_min_is_clamped(self) -> None:
        c = fresh_armed_controller()
        applied = c.set_current_setpoint(-5.0)
        assert applied == c.config.current_setpoint_min_a

    def test_setpoint_zero_keeps_state_armed(self) -> None:
        c = fresh_armed_controller()
        c.set_current_setpoint(0.0)
        assert c.state == PowerSupplyState.ARMED

    def test_setpoint_ignored_in_idle_state(self) -> None:
        c = fresh_idle_controller()
        applied = c.set_current_setpoint(25.0)
        assert applied == 25.0  # value clamped + returned
        assert c.state == PowerSupplyState.IDLE  # but not applied
        cmd = c.tick(0.0, 0.0, 25.0)
        assert cmd == 0.0

    def test_setpoint_ignored_in_tripped_state(self) -> None:
        c = fresh_running_controller(10.0)
        c.tick(200.0, 20.0, 25.0)  # trip on power
        assert c.state == PowerSupplyState.TRIPPED
        c.set_current_setpoint(5.0)
        cmd = c.tick(0.0, 0.0, 25.0)
        assert cmd == 0.0

    def test_rejects_nan(self) -> None:
        c = fresh_armed_controller()
        with pytest.raises(ValueError, match="finite"):
            c.set_current_setpoint(float("nan"))

    def test_rejects_infinity(self) -> None:
        c = fresh_armed_controller()
        with pytest.raises(ValueError, match="finite"):
            c.set_current_setpoint(float("inf"))
        with pytest.raises(ValueError, match="finite"):
            c.set_current_setpoint(float("-inf"))

    def test_rejects_non_numeric(self) -> None:
        c = fresh_armed_controller()
        with pytest.raises(TypeError):
            c.set_current_setpoint("10")
        with pytest.raises(TypeError):
            c.set_current_setpoint(None)

    def test_rejects_bool(self) -> None:
        # Bool is technically a subclass of int. Explicit catch keeps the
        # caller honest.
        c = fresh_armed_controller()
        with pytest.raises(TypeError):
            c.set_current_setpoint(True)


# --------------------------------------------------------------------------
# Power setpoint — derives current from measured voltage
# --------------------------------------------------------------------------


class TestPowerSetpoint:
    def test_power_setpoint_derives_current_from_voltage(self) -> None:
        c = fresh_armed_controller()
        # Establish a measured voltage via tick before setting power.
        c.tick(measured_current_a=0.0, measured_voltage_v=20.0, measured_temp_c=25.0)
        achievable = c.set_power_setpoint(200.0)
        # 200 W / 20 V = 10 A target. Within bounds.
        assert achievable == pytest.approx(200.0)
        # In POWER mode now
        assert c.mode == PowerSupplyMode.POWER

    def test_power_setpoint_clamped_by_current_max(self) -> None:
        c = fresh_armed_controller()
        c.tick(0.0, 10.0, 25.0)  # V = 10 V
        # 1000 W / 10 V = 100 A target, clamped to current_setpoint_max_a (50)
        achievable = c.set_power_setpoint(1000.0)
        # achievable = 50 A * 10 V = 500 W
        assert achievable == pytest.approx(500.0)

    def test_power_setpoint_rejected_when_voltage_too_low(self) -> None:
        c = fresh_armed_controller()
        c.tick(0.0, 0.0, 25.0)  # V = 0
        achievable = c.set_power_setpoint(100.0)
        assert achievable == 0.0
        # Mode unchanged
        assert c.state == PowerSupplyState.ARMED

    def test_power_setpoint_recomputed_each_tick(self) -> None:
        c = fresh_armed_controller()
        c.tick(0.0, 20.0, 25.0, now=0.0)
        c.set_power_setpoint(200.0)  # → 10 A at 20 V
        # Slew needs a couple ticks with elapsed time to actually reach
        # 20 % from a freshly-armed (0 %) starting point.
        c.tick(
            measured_current_a=10.0,
            measured_voltage_v=10.0,
            measured_temp_c=25.0,
            now=0.1,
        )
        cmd1 = c.tick(
            measured_current_a=10.0,
            measured_voltage_v=10.0,
            measured_temp_c=25.0,
            now=100.0,
        )
        # Should re-target: 200 / 10 = 20 A. cmd1 should be 20 % of full-scale (100 A)
        assert cmd1 == pytest.approx(20.0)

    def test_rejects_negative_power(self) -> None:
        c = fresh_armed_controller()
        c.tick(0.0, 20.0, 25.0)
        with pytest.raises(ValueError, match="non-negative"):
            c.set_power_setpoint(-100.0)

    def test_rejects_nan_inf_power(self) -> None:
        c = fresh_armed_controller()
        c.tick(0.0, 20.0, 25.0)
        with pytest.raises(ValueError, match="finite"):
            c.set_power_setpoint(float("nan"))
        with pytest.raises(ValueError, match="finite"):
            c.set_power_setpoint(float("inf"))

    def test_rejects_non_numeric_power(self) -> None:
        c = fresh_armed_controller()
        c.tick(0.0, 20.0, 25.0)
        with pytest.raises(TypeError):
            c.set_power_setpoint("100")


# --------------------------------------------------------------------------
# Trip logic — HH_POWER and HH_TEMP
# --------------------------------------------------------------------------


class TestTrips:
    def test_hh_power_trips_when_measured_exceeds_threshold(self) -> None:
        c = PowerSupplyController(PowerSupplyConfig(power_alarm_max_w=500.0))
        c.set_permissive(True)
        c.set_current_setpoint(10.0)
        # P = V * I = 30 * 20 = 600 W, exceeds 500
        cmd = c.tick(
            measured_current_a=20.0, measured_voltage_v=30.0, measured_temp_c=25.0
        )
        assert cmd == 0.0  # output zeroed
        assert c.state == PowerSupplyState.TRIPPED
        assert "HH_POWER" in c.trips

    def test_hh_power_does_not_trip_at_threshold_minus_epsilon(self) -> None:
        c = PowerSupplyController(PowerSupplyConfig(power_alarm_max_w=500.0))
        c.set_permissive(True)
        c.set_current_setpoint(10.0)
        c.tick(measured_current_a=10.0, measured_voltage_v=49.0, measured_temp_c=25.0)
        # P = 490 W < 500 → no trip
        assert c.state == PowerSupplyState.RUNNING
        assert c.trips == []

    def test_hh_temp_trips_above_threshold(self) -> None:
        c = PowerSupplyController(PowerSupplyConfig(temp_alarm_max_c=1200.0))
        c.set_permissive(True)
        c.set_current_setpoint(10.0)
        cmd = c.tick(
            measured_current_a=10.0, measured_voltage_v=5.0, measured_temp_c=1500.0
        )
        assert cmd == 0.0
        assert c.state == PowerSupplyState.TRIPPED
        assert "HH_TEMP" in c.trips

    def test_both_trips_can_latch_simultaneously(self) -> None:
        c = PowerSupplyController(
            PowerSupplyConfig(power_alarm_max_w=100.0, temp_alarm_max_c=100.0)
        )
        c.set_permissive(True)
        c.set_current_setpoint(10.0)
        c.tick(measured_current_a=50.0, measured_voltage_v=10.0, measured_temp_c=200.0)
        assert c.state == PowerSupplyState.TRIPPED
        assert "HH_POWER" in c.trips
        assert "HH_TEMP" in c.trips

    def test_trip_latches_through_safe_subsequent_tick(self) -> None:
        c = PowerSupplyController(PowerSupplyConfig(power_alarm_max_w=500.0))
        c.set_permissive(True)
        c.set_current_setpoint(10.0)
        c.tick(measured_current_a=20.0, measured_voltage_v=30.0, measured_temp_c=25.0)
        assert c.state == PowerSupplyState.TRIPPED
        # Now signal returns to safe band — trip MUST stay latched
        c.tick(measured_current_a=1.0, measured_voltage_v=1.0, measured_temp_c=25.0)
        assert c.state == PowerSupplyState.TRIPPED
        assert "HH_POWER" in c.trips

    def test_acknowledge_trip_clears_trips_and_state(self) -> None:
        c = fresh_running_controller(10.0)
        c.tick(measured_current_a=200.0, measured_voltage_v=20.0, measured_temp_c=25.0)
        assert c.state == PowerSupplyState.TRIPPED
        cleared = c.acknowledge_trip()
        assert cleared is True
        assert c.trips == []
        # Permissive was on; ack returns to ARMED, not IDLE
        assert c.state == PowerSupplyState.ARMED

    def test_acknowledge_with_permissive_off_returns_to_idle(self) -> None:
        c = fresh_running_controller(10.0)
        c.tick(200.0, 20.0, 25.0)
        c.set_permissive(False)
        c.acknowledge_trip()
        assert c.state == PowerSupplyState.IDLE

    def test_acknowledge_with_no_trip_returns_false(self) -> None:
        c = fresh_armed_controller()
        assert c.acknowledge_trip() is False


# --------------------------------------------------------------------------
# tick() output command behavior
# --------------------------------------------------------------------------


class TestTickOutput:
    def test_tick_returns_zero_in_idle(self) -> None:
        c = fresh_idle_controller()
        assert c.tick(0.0, 0.0, 25.0) == 0.0

    def test_tick_returns_zero_in_armed(self) -> None:
        c = fresh_armed_controller()
        assert c.tick(0.0, 0.0, 25.0) == 0.0

    def test_tick_returns_zero_when_permissive_off_after_running(self) -> None:
        c = fresh_running_controller(10.0)
        c.set_permissive(False)
        assert c.tick(0.0, 0.0, 25.0) == 0.0

    def test_tick_returns_scaled_percent_when_running(self) -> None:
        # Default current_full_scale_a = 100 A; setpoint 25 A → 25 %
        c = fresh_armed_controller()
        c.set_current_setpoint(25.0)
        # Two ticks with large dt so slew limiter doesn't gate the result.
        c.tick(0.0, 0.0, 25.0, now=0.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=100.0)
        assert cmd == pytest.approx(25.0)

    def test_tick_clamps_percent_to_100(self) -> None:
        # Construct so setpoint can equal full scale.
        cfg = PowerSupplyConfig(
            current_full_scale_a=100.0,
            current_setpoint_max_a=100.0,
        )
        c = PowerSupplyController(cfg)
        c.set_permissive(True)
        c.set_current_setpoint(100.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=100.0)
        assert cmd == pytest.approx(100.0)

    def test_tick_handles_nan_inputs_safely(self) -> None:
        c = fresh_running_controller(10.0)
        c.tick(
            measured_current_a=float("nan"),
            measured_voltage_v=float("inf"),
            measured_temp_c=float("nan"),
            now=0.0,
        )
        cmd = c.tick(
            measured_current_a=float("nan"),
            measured_voltage_v=float("inf"),
            measured_temp_c=float("nan"),
            now=100.0,
        )
        # Bad inputs → treated as 0 → no trip, command tracks setpoint
        assert cmd > 0.0
        assert c.state == PowerSupplyState.RUNNING


# --------------------------------------------------------------------------
# Slew-rate limiter — slow-start ramp on increases, instant on decreases
# --------------------------------------------------------------------------


class TestSlewRate:
    """Slow-start ramp behavior:
    - Increases in commanded output are clamped to
      `setpoint_ramp_rate_pct_per_s * dt`.
    - Decreases pass through immediately (operator can always pull down).
    - Trip / permissive-off / IDLE / ARMED all force output to zero
      instantly, bypassing the ramp.
    """

    def test_default_ramp_rate_is_5_percent_per_second(self) -> None:
        cfg = PowerSupplyConfig()
        assert cfg.setpoint_ramp_rate_pct_per_s == 5.0

    def test_ramp_rate_must_be_positive(self) -> None:
        with pytest.raises(Exception, match="greater than 0"):
            PowerSupplyConfig(setpoint_ramp_rate_pct_per_s=0.0)
        with pytest.raises(Exception, match="greater than 0"):
            PowerSupplyConfig(setpoint_ramp_rate_pct_per_s=-1.0)

    def test_first_tick_after_setpoint_is_zero_dt_so_no_movement(self) -> None:
        """The very first tick into RUNNING establishes the time baseline;
        with dt=0 the slew can't advance, so output stays at 0. This is the
        slow-start invariant — no snap-up to the commanded value."""
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)  # target 50 %
        cmd = c.tick(0.0, 0.0, 25.0, now=10.0)
        assert cmd == pytest.approx(0.0)

    def test_ramp_advances_at_configured_rate(self) -> None:
        """At 5 %/s, one second of elapsed time should advance the command
        by exactly 5 percentage points from its current value."""
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)  # baseline t=0
        cmd1 = c.tick(0.0, 0.0, 25.0, now=1.0)
        assert cmd1 == pytest.approx(5.0)  # +5 %/s * 1 s
        cmd2 = c.tick(0.0, 0.0, 25.0, now=3.0)
        assert cmd2 == pytest.approx(15.0)  # +10 % over 2 more seconds

    def test_ramp_settles_at_target_when_given_enough_time(self) -> None:
        c = fresh_armed_controller()
        c.set_current_setpoint(30.0)  # target 30 %
        c.tick(0.0, 0.0, 25.0, now=0.0)
        # 30 % / (5 %/s) = 6 s. Give 10 s just to be sure.
        cmd = c.tick(0.0, 0.0, 25.0, now=10.0)
        assert cmd == pytest.approx(30.0)

    def test_setpoint_decrease_is_instant_no_ramp(self) -> None:
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)
        c.tick(0.0, 0.0, 25.0, now=10.0)  # settled at 50 %
        # Operator pulls down to 5 %
        c.set_current_setpoint(5.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=10.01)  # 10 ms later
        assert cmd == pytest.approx(5.0)

    def test_setpoint_to_zero_is_instant(self) -> None:
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)
        c.tick(0.0, 0.0, 25.0, now=10.0)  # at 50 %
        c.set_current_setpoint(0.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=10.001)
        assert cmd == pytest.approx(0.0)

    def test_permissive_off_drops_output_to_zero_without_ramp(self) -> None:
        """Operator hitting the permissive must be a one-tick kill."""
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)
        c.tick(0.0, 0.0, 25.0, now=10.0)  # at 50 %
        c.set_permissive(False)
        cmd = c.tick(0.0, 0.0, 25.0, now=10.001)
        assert cmd == pytest.approx(0.0)

    def test_trip_drops_output_to_zero_without_ramp(self) -> None:
        """A trip while ramping must be a one-tick kill."""
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)
        c.tick(
            measured_current_a=0.0,
            measured_voltage_v=10.0,
            measured_temp_c=25.0,
            now=0.0,
        )
        c.tick(
            measured_current_a=0.0,
            measured_voltage_v=10.0,
            measured_temp_c=25.0,
            now=2.0,
        )
        # Now drive a trip with the very next tick
        cmd = c.tick(
            measured_current_a=200.0,
            measured_voltage_v=20.0,
            measured_temp_c=25.0,
            now=2.001,
        )
        assert cmd == pytest.approx(0.0)
        assert c.state == PowerSupplyState.TRIPPED

    def test_custom_ramp_rate_respected(self) -> None:
        """Setting a 1 %/s rate should produce a much slower ramp."""
        cfg = PowerSupplyConfig(setpoint_ramp_rate_pct_per_s=1.0)
        c = PowerSupplyController(cfg)
        c.set_permissive(True)
        c.set_current_setpoint(20.0)  # target 20 %
        c.tick(0.0, 0.0, 25.0, now=0.0)
        cmd1 = c.tick(0.0, 0.0, 25.0, now=5.0)
        assert cmd1 == pytest.approx(5.0)  # 1 %/s * 5 s

    def test_ramp_state_resets_after_permissive_cycle(self) -> None:
        """Cycling permissive off->on should restart the ramp from zero."""
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)
        c.tick(0.0, 0.0, 25.0, now=10.0)  # at 50 %
        # Operator hits permissive off then back on
        c.set_permissive(False)
        c.tick(0.0, 0.0, 25.0, now=10.001)  # zeroed
        c.set_permissive(True)
        c.set_current_setpoint(50.0)
        # Fresh ramp baseline at t=20; next tick at +1 s should be 5 %, not 50 %
        c.tick(0.0, 0.0, 25.0, now=20.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=21.0)
        assert cmd == pytest.approx(5.0)

    def test_step_up_during_existing_ramp_continues_at_rate(self) -> None:
        """Bumping the setpoint up mid-ramp must not snap output upward —
        the slew limit still applies from wherever we are now."""
        c = fresh_armed_controller()
        c.set_current_setpoint(30.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)
        c.tick(0.0, 0.0, 25.0, now=2.0)  # at 10 %
        c.set_current_setpoint(40.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=2.5)  # 0.5 s later
        # From 10 %, +5 %/s * 0.5 s = 12.5 %, NOT a jump to 40 %
        assert cmd == pytest.approx(12.5)
