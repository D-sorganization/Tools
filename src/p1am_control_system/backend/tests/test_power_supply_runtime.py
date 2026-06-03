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

import pytest
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
        cmd = c.tick(0.0, 0.0, 25.0)
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
            c.set_current_setpoint("10")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            c.set_current_setpoint(None)  # type: ignore[arg-type]

    def test_rejects_bool(self) -> None:
        # Bool is technically a subclass of int. Explicit catch keeps the
        # caller honest.
        c = fresh_armed_controller()
        with pytest.raises(TypeError):
            c.set_current_setpoint(True)  # type: ignore[arg-type]


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
        c.tick(0.0, 20.0, 25.0)
        c.set_power_setpoint(200.0)  # → 10 A at 20 V
        # Now voltage drops
        cmd1 = c.tick(
            measured_current_a=10.0, measured_voltage_v=10.0, measured_temp_c=25.0
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
            c.set_power_setpoint("100")  # type: ignore[arg-type]


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
        cmd = c.tick(0.0, 0.0, 25.0)
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
        cmd = c.tick(0.0, 0.0, 25.0)
        assert cmd == 100.0

    def test_tick_handles_nan_inputs_safely(self) -> None:
        c = fresh_running_controller(10.0)
        cmd = c.tick(
            measured_current_a=float("nan"),
            measured_voltage_v=float("inf"),
            measured_temp_c=float("nan"),
        )
        # Bad inputs → treated as 0 → no trip, command tracks setpoint
        assert cmd > 0.0
        assert c.state == PowerSupplyState.RUNNING
