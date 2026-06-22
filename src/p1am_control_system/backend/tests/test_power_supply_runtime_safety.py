"""Runtime safety tests for PowerSupplyController.

`test_power_supply_runtime.py` covers setpoint behavior. This file covers:
    - HH_POWER and HH_TEMP trip latching across safe subsequent ticks
    - Trip acknowledgement + state transitions
    - tick() output guarantees in IDLE / ARMED / TRIPPED / no-permissive states
    - Output clamp behavior and slew-rate limiting
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _power_supply_helpers import (
    fresh_armed_controller,
    fresh_armed_controller_unclamped,
    fresh_idle_controller,
    fresh_running_controller,
    test_config,
)
from power_supply import PowerSupplyConfig, PowerSupplyController, PowerSupplyState

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
        # P = 490 W < 500 -> no trip
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
        # Now signal returns to safe band; trip must stay latched.
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
        # Default current_full_scale_a = 100 A; setpoint 25 A -> 25 %.
        # Unclamped so the 20 % output clamp doesn't cap the 25 % result.
        c = fresh_armed_controller_unclamped()
        c.set_current_setpoint(25.0)
        # Two ticks with large dt so slew limiter doesn't gate the result.
        c.tick(0.0, 0.0, 25.0, now=0.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=100.0)
        assert cmd == pytest.approx(25.0)

    def test_tick_clamps_percent_to_100(self) -> None:
        # Construct so setpoint can equal full scale, with the output clamp
        # opened to 100 % so this isolates the 100 % ceiling behavior.
        cfg = PowerSupplyConfig(
            current_full_scale_a=100.0,
            current_setpoint_max_a=100.0,
            output_clamp_percent=100.0,
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
        # Bad inputs -> treated as 0 -> no trip, command tracks setpoint
        assert cmd > 0.0
        assert c.state == PowerSupplyState.RUNNING


# --------------------------------------------------------------------------
# Output clamp — operator safety cap on commanded output (live-current limit)
# --------------------------------------------------------------------------


class TestOutputClamp:
    """The output clamp hard-caps the commanded AO percent regardless of how
    the current setpoint scales. It is the operator's safety limit when
    testing into live current. Default 20 %, user-adjustable, applied before
    the slew limiter so the ramp settles at the clamp."""

    @staticmethod
    def _settle(c: PowerSupplyController) -> float:
        """Two ticks with a huge dt so the slew limiter never gates the
        result; returns the settled command percent."""
        c.tick(0.0, 0.0, 25.0, now=0.0)
        return float(c.tick(0.0, 0.0, 25.0, now=1000.0))

    def test_command_capped_at_clamp_when_setpoint_higher(self) -> None:
        # Default clamp 20 %; setpoint 50 A -> 50 % raw, capped to 20 %.
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)
        cmd = self._settle(c)
        assert cmd == pytest.approx(20.0)
        assert c.status().output_clamped is True

    def test_command_unaffected_when_below_clamp(self) -> None:
        # Setpoint 10 A -> 10 % raw, under the 20 % clamp -> passes through.
        c = fresh_armed_controller()
        c.set_current_setpoint(10.0)
        cmd = self._settle(c)
        assert cmd == pytest.approx(10.0)
        assert c.status().output_clamped is False

    def test_custom_clamp_value_is_enforced(self) -> None:
        cfg = test_config(output_clamp_percent=35.0)  # 100 A full scale
        c = PowerSupplyController(cfg)
        c.set_permissive(True)
        c.set_current_setpoint(50.0)  # 50 % raw
        cmd = self._settle(c)
        assert cmd == pytest.approx(35.0)
        assert c.status().output_clamped is True

    def test_lowering_clamp_takes_effect_immediately(self) -> None:
        # Settle at the default 20 % clamp, then lower it to 10 %. A decrease
        # passes through the slew limiter instantly.
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)
        assert self._settle(c) == pytest.approx(20.0)
        c.update_config(PowerSupplyConfig(output_clamp_percent=10.0))
        cmd = c.tick(0.0, 0.0, 25.0, now=1001.0)
        assert cmd == pytest.approx(10.0)
        assert c.status().output_clamped is True

    def test_clamp_status_fields_reported(self) -> None:
        c = PowerSupplyController(PowerSupplyConfig(output_clamp_percent=15.0))
        status = c.status()
        assert status.output_clamp_percent == pytest.approx(15.0)
        assert status.output_clamped is False

    def test_force_zero_clears_clamped_flag(self) -> None:
        c = fresh_armed_controller()
        c.set_current_setpoint(50.0)
        self._settle(c)
        assert c.status().output_clamped is True
        c.set_permissive(False)
        assert c.tick(0.0, 0.0, 25.0, now=1001.0) == 0.0
        assert c.status().output_clamped is False


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
        slow-start invariant; no snap-up to the commanded value."""
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
        c = fresh_armed_controller_unclamped()  # 30 % target exceeds 20 % clamp
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
        """Bumping the setpoint up mid-ramp must not snap output upward;
        the slew limit still applies from wherever we are now."""
        c = fresh_armed_controller()
        c.set_current_setpoint(30.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)
        c.tick(0.0, 0.0, 25.0, now=2.0)  # at 10 %
        c.set_current_setpoint(40.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=2.5)  # 0.5 s later
        # From 10 %, +5 %/s * 0.5 s = 12.5 %, not a jump to 40 %
        assert cmd == pytest.approx(12.5)


# --------------------------------------------------------------------------
# E-stop — latched kill switch (software half)
# --------------------------------------------------------------------------


class TestEstop:
    """engage_estop() latches output to zero and disarms; the latch survives
    re-arm attempts and setpoint commands until clear_estop() is called."""

    def test_engage_forces_output_zero_from_running(self) -> None:
        c = fresh_running_controller(40.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)
        c.tick(0.0, 0.0, 25.0, now=100.0)  # ramped up
        c.engage_estop()
        cmd = c.tick(0.0, 0.0, 25.0, now=200.0)
        assert cmd == 0.0
        assert c.estopped is True
        assert c.state == PowerSupplyState.IDLE
        assert c.permissive is False

    def test_cannot_arm_while_estopped(self) -> None:
        c = fresh_running_controller(40.0)
        c.engage_estop()
        c.set_permissive(True)  # ignored
        assert c.permissive is False
        assert c.state == PowerSupplyState.IDLE
        assert c.tick(0.0, 0.0, 25.0, now=300.0) == 0.0

    def test_setpoint_rejected_while_estopped(self) -> None:
        c = fresh_running_controller(40.0)
        c.engage_estop()
        assert c.set_current_setpoint(50.0) == 0.0
        assert c.set_power_setpoint(100.0) == 0.0
        assert c.tick(0.0, 0.0, 25.0, now=400.0) == 0.0

    def test_clear_releases_latch_and_requires_rearm(self) -> None:
        c = fresh_running_controller(40.0)
        c.engage_estop()
        c.clear_estop()
        assert c.estopped is False
        assert c.permissive is False  # must re-arm explicitly
        assert c.state == PowerSupplyState.IDLE
        # After re-arm + setpoint, output flows again.
        c.set_permissive(True)
        c.set_current_setpoint(10.0)
        c.tick(0.0, 0.0, 25.0, now=0.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=100.0)
        assert cmd == pytest.approx(10.0)

    def test_clear_when_not_estopped_is_noop(self) -> None:
        c = fresh_running_controller(10.0)
        c.clear_estop()
        assert c.estopped is False
        assert c.state == PowerSupplyState.RUNNING
