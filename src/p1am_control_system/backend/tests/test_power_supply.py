"""Unit tests for the PowerSupplyController and its config model.

Covers:
    - Pydantic config validation (field constraints, cross-field invariants)
    - State machine transitions across all four states
    - Current setpoint clamping (fat-finger protection)
    - Power setpoint derivation from measured voltage
    - HH_POWER and HH_TEMP trip latching
    - Trip acknowledge / reset
    - Permissive on/off semantics
    - Mode switching between CURRENT and POWER
    - Type/value validation rejecting bool, NaN, infinity

Designed to run with no PLC connection. Every behavior the operator might
encounter (out-of-range setpoint, trip during runtime, trip with permissive
off, etc.) is asserted explicitly.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
from pydantic import ValidationError

# Allow running from this file's directory (backend/tests/) by adding the
# backend dir to sys.path so `import power_supply` resolves.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from power_supply import (  # noqa: E402  (path setup above must run first)
    PowerSupplyConfig,
    PowerSupplyController,
    PowerSupplyMode,
    PowerSupplyState,
)

# --------------------------------------------------------------------------
# PowerSupplyConfig validation
# --------------------------------------------------------------------------


class TestPowerSupplyConfigValidation:
    def test_defaults_construct(self) -> None:
        config = PowerSupplyConfig()
        assert config.current_full_scale_a == 100.0
        assert config.voltage_full_scale_v == 50.0
        assert config.current_setpoint_min_a == 0.0
        assert config.current_setpoint_max_a == 50.0

    def test_full_scale_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PowerSupplyConfig(current_full_scale_a=0.0)
        with pytest.raises(ValidationError):
            PowerSupplyConfig(voltage_full_scale_v=-1.0)

    def test_setpoint_min_cannot_exceed_max(self) -> None:
        with pytest.raises(ValidationError):
            PowerSupplyConfig(current_setpoint_min_a=60.0, current_setpoint_max_a=50.0)

    def test_setpoint_min_equal_to_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PowerSupplyConfig(current_setpoint_min_a=50.0, current_setpoint_max_a=50.0)

    def test_setpoint_max_cannot_exceed_full_scale(self) -> None:
        with pytest.raises(ValidationError):
            PowerSupplyConfig(
                current_full_scale_a=100.0,
                current_setpoint_max_a=150.0,
            )

    def test_alarm_thresholds_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PowerSupplyConfig(power_alarm_max_w=0.0)
        with pytest.raises(ValidationError):
            PowerSupplyConfig(temp_alarm_max_c=-10.0)


# --------------------------------------------------------------------------
# Constructor + initial state
# --------------------------------------------------------------------------


def _ctrl() -> PowerSupplyController:
    return PowerSupplyController(PowerSupplyConfig())


def _armed_ctrl() -> PowerSupplyController:
    c = _ctrl()
    c.set_permissive(True)
    return c


def _running_ctrl(sp_a: float = 10.0) -> PowerSupplyController:
    c = _armed_ctrl()
    c.set_current_setpoint(sp_a)
    return c


class TestController_Init:
    def test_initial_state_is_idle(self) -> None:
        c = _ctrl()
        assert c.state == PowerSupplyState.IDLE
        assert c.permissive is False
        assert c.trips == []

    def test_rejects_non_config_argument(self) -> None:
        with pytest.raises(TypeError):
            PowerSupplyController("not a config")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            PowerSupplyController({"current_full_scale_a": 10.0})  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Permissive transitions
# --------------------------------------------------------------------------


class TestPermissive:
    def test_idle_to_armed_when_permissive_on(self) -> None:
        c = _ctrl()
        c.set_permissive(True)
        assert c.state == PowerSupplyState.ARMED
        assert c.permissive is True

    def test_armed_to_idle_when_permissive_off(self) -> None:
        c = _armed_ctrl()
        c.set_permissive(False)
        assert c.state == PowerSupplyState.IDLE

    def test_running_to_idle_when_permissive_off(self) -> None:
        c = _running_ctrl(10.0)
        assert c.state == PowerSupplyState.RUNNING
        c.set_permissive(False)
        assert c.state == PowerSupplyState.IDLE

    def test_permissive_off_clears_setpoint(self) -> None:
        c = _running_ctrl(15.0)
        c.set_permissive(False)
        # Re-arming gives a clean state at 0 setpoint
        c.set_permissive(True)
        assert c.state == PowerSupplyState.ARMED
        cmd = c.tick(0.0, 0.0, 25.0)
        assert cmd == 0.0

    def test_permissive_change_does_not_clear_trip(self) -> None:
        c = _running_ctrl(10.0)
        # Drive a trip via measured power
        c.tick(measured_current_a=200.0, measured_voltage_v=20.0, measured_temp_c=25.0)
        assert c.state == PowerSupplyState.TRIPPED
        c.set_permissive(False)
        assert c.state == PowerSupplyState.TRIPPED
        c.set_permissive(True)
        assert c.state == PowerSupplyState.TRIPPED

    def test_rejects_non_bool_permissive(self) -> None:
        c = _ctrl()
        with pytest.raises(TypeError):
            c.set_permissive(1)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            c.set_permissive("on")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Current setpoint — fat-finger protection
# --------------------------------------------------------------------------


class TestCurrentSetpoint:
    def test_in_band_setpoint_applied(self) -> None:
        c = _armed_ctrl()
        applied = c.set_current_setpoint(25.0)
        assert applied == 25.0
        assert c.state == PowerSupplyState.RUNNING

    def test_setpoint_above_max_is_clamped(self) -> None:
        c = _armed_ctrl()
        applied = c.set_current_setpoint(9999.0)
        assert applied == c.config.current_setpoint_max_a
        cmd = c.tick(0.0, 0.0, 25.0)
        assert cmd == pytest.approx(
            100.0 * c.config.current_setpoint_max_a / c.config.current_full_scale_a
        )

    def test_setpoint_below_min_is_clamped(self) -> None:
        c = _armed_ctrl()
        applied = c.set_current_setpoint(-5.0)
        assert applied == c.config.current_setpoint_min_a

    def test_setpoint_zero_keeps_state_armed(self) -> None:
        c = _armed_ctrl()
        c.set_current_setpoint(0.0)
        assert c.state == PowerSupplyState.ARMED

    def test_setpoint_ignored_in_idle_state(self) -> None:
        c = _ctrl()
        applied = c.set_current_setpoint(25.0)
        assert applied == 25.0  # value clamped + returned
        assert c.state == PowerSupplyState.IDLE  # but not applied
        cmd = c.tick(0.0, 0.0, 25.0)
        assert cmd == 0.0

    def test_setpoint_ignored_in_tripped_state(self) -> None:
        c = _running_ctrl(10.0)
        c.tick(200.0, 20.0, 25.0)  # trip on power
        assert c.state == PowerSupplyState.TRIPPED
        c.set_current_setpoint(5.0)
        cmd = c.tick(0.0, 0.0, 25.0)
        assert cmd == 0.0

    def test_rejects_nan(self) -> None:
        c = _armed_ctrl()
        with pytest.raises(ValueError):
            c.set_current_setpoint(float("nan"))

    def test_rejects_infinity(self) -> None:
        c = _armed_ctrl()
        with pytest.raises(ValueError):
            c.set_current_setpoint(float("inf"))
        with pytest.raises(ValueError):
            c.set_current_setpoint(float("-inf"))

    def test_rejects_non_numeric(self) -> None:
        c = _armed_ctrl()
        with pytest.raises(TypeError):
            c.set_current_setpoint("10")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            c.set_current_setpoint(None)  # type: ignore[arg-type]

    def test_rejects_bool(self) -> None:
        # Bool is technically a subclass of int. Explicit catch keeps the
        # caller honest.
        c = _armed_ctrl()
        with pytest.raises(TypeError):
            c.set_current_setpoint(True)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Power setpoint — derives current from measured voltage
# --------------------------------------------------------------------------


class TestPowerSetpoint:
    def test_power_setpoint_derives_current_from_voltage(self) -> None:
        c = _armed_ctrl()
        # Establish a measured voltage via tick before setting power.
        c.tick(measured_current_a=0.0, measured_voltage_v=20.0, measured_temp_c=25.0)
        achievable = c.set_power_setpoint(200.0)
        # 200 W / 20 V = 10 A target. Within bounds.
        assert achievable == pytest.approx(200.0)
        # In POWER mode now
        assert c.mode == PowerSupplyMode.POWER

    def test_power_setpoint_clamped_by_current_max(self) -> None:
        c = _armed_ctrl()
        c.tick(0.0, 10.0, 25.0)  # V = 10 V
        # 1000 W / 10 V = 100 A target, clamped to current_setpoint_max_a (50)
        achievable = c.set_power_setpoint(1000.0)
        # achievable = 50 A * 10 V = 500 W
        assert achievable == pytest.approx(500.0)

    def test_power_setpoint_rejected_when_voltage_too_low(self) -> None:
        c = _armed_ctrl()
        c.tick(0.0, 0.0, 25.0)  # V = 0
        achievable = c.set_power_setpoint(100.0)
        assert achievable == 0.0
        # Mode unchanged
        assert c.state == PowerSupplyState.ARMED

    def test_power_setpoint_recomputed_each_tick(self) -> None:
        c = _armed_ctrl()
        c.tick(0.0, 20.0, 25.0)
        c.set_power_setpoint(200.0)  # → 10 A at 20 V
        # Now voltage drops
        cmd1 = c.tick(
            measured_current_a=10.0, measured_voltage_v=10.0, measured_temp_c=25.0
        )
        # Should re-target: 200 / 10 = 20 A. cmd1 should be 20 % of full-scale (100 A)
        assert cmd1 == pytest.approx(20.0)

    def test_rejects_negative_power(self) -> None:
        c = _armed_ctrl()
        c.tick(0.0, 20.0, 25.0)
        with pytest.raises(ValueError):
            c.set_power_setpoint(-100.0)

    def test_rejects_nan_inf_power(self) -> None:
        c = _armed_ctrl()
        c.tick(0.0, 20.0, 25.0)
        with pytest.raises(ValueError):
            c.set_power_setpoint(float("nan"))
        with pytest.raises(ValueError):
            c.set_power_setpoint(float("inf"))

    def test_rejects_non_numeric_power(self) -> None:
        c = _armed_ctrl()
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
        c = _running_ctrl(10.0)
        c.tick(measured_current_a=200.0, measured_voltage_v=20.0, measured_temp_c=25.0)
        assert c.state == PowerSupplyState.TRIPPED
        cleared = c.acknowledge_trip()
        assert cleared is True
        assert c.trips == []
        # Permissive was on; ack returns to ARMED, not IDLE
        assert c.state == PowerSupplyState.ARMED

    def test_acknowledge_with_permissive_off_returns_to_idle(self) -> None:
        c = _running_ctrl(10.0)
        c.tick(200.0, 20.0, 25.0)
        c.set_permissive(False)
        c.acknowledge_trip()
        assert c.state == PowerSupplyState.IDLE

    def test_acknowledge_with_no_trip_returns_false(self) -> None:
        c = _armed_ctrl()
        assert c.acknowledge_trip() is False


# --------------------------------------------------------------------------
# tick() output command behavior
# --------------------------------------------------------------------------


class TestTickOutput:
    def test_tick_returns_zero_in_idle(self) -> None:
        c = _ctrl()
        assert c.tick(0.0, 0.0, 25.0) == 0.0

    def test_tick_returns_zero_in_armed(self) -> None:
        c = _armed_ctrl()
        assert c.tick(0.0, 0.0, 25.0) == 0.0

    def test_tick_returns_zero_when_permissive_off_after_running(self) -> None:
        c = _running_ctrl(10.0)
        c.set_permissive(False)
        assert c.tick(0.0, 0.0, 25.0) == 0.0

    def test_tick_returns_scaled_percent_when_running(self) -> None:
        # Default current_full_scale_a = 100 A; setpoint 25 A → 25 %
        c = _armed_ctrl()
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
        c = _running_ctrl(10.0)
        cmd = c.tick(
            measured_current_a=float("nan"),
            measured_voltage_v=float("inf"),
            measured_temp_c=float("nan"),
        )
        # Bad inputs → treated as 0 → no trip, command tracks setpoint
        assert cmd > 0.0
        assert c.state == PowerSupplyState.RUNNING


# --------------------------------------------------------------------------
# Config update at runtime
# --------------------------------------------------------------------------


class TestUpdateConfig:
    def test_update_config_re_clamps_existing_setpoint(self) -> None:
        c = _armed_ctrl()
        c.set_current_setpoint(40.0)
        # New config with stricter max → setpoint should re-clamp.
        new_cfg = PowerSupplyConfig(
            current_full_scale_a=100.0,
            current_setpoint_max_a=20.0,
        )
        c.update_config(new_cfg)
        cmd = c.tick(0.0, 0.0, 25.0)
        # Setpoint should now be 20 A → 20 % of full scale (100 A)
        assert cmd == pytest.approx(20.0)

    def test_update_config_type_check(self) -> None:
        c = _ctrl()
        with pytest.raises(TypeError):
            c.update_config({"current_full_scale_a": 50.0})  # type: ignore[arg-type]

    def test_invalid_config_rejected_at_construction(self) -> None:
        # update_config can't take an invalid config because invalid
        # configs cannot be constructed in the first place.
        with pytest.raises(ValidationError):
            PowerSupplyConfig(
                current_full_scale_a=10.0,
                current_setpoint_max_a=50.0,
            )


# --------------------------------------------------------------------------
# Mode switching
# --------------------------------------------------------------------------


class TestModeSwitching:
    def test_current_setpoint_switches_mode_to_current(self) -> None:
        c = _armed_ctrl()
        c.tick(0.0, 20.0, 25.0)
        c.set_power_setpoint(200.0)
        assert c.mode == PowerSupplyMode.POWER
        c.set_current_setpoint(15.0)
        assert c.mode == PowerSupplyMode.CURRENT

    def test_power_setpoint_switches_mode_to_power(self) -> None:
        c = _armed_ctrl()
        c.set_current_setpoint(15.0)
        assert c.mode == PowerSupplyMode.CURRENT
        c.tick(0.0, 20.0, 25.0)
        c.set_power_setpoint(200.0)
        assert c.mode == PowerSupplyMode.POWER


# --------------------------------------------------------------------------
# Status snapshot
# --------------------------------------------------------------------------


class TestStatus:
    def test_status_reports_current_state(self) -> None:
        c = _running_ctrl(10.0)
        c.tick(measured_current_a=5.0, measured_voltage_v=10.0, measured_temp_c=30.0)
        s = c.status()
        assert s.state == PowerSupplyState.RUNNING
        assert s.permissive is True
        assert s.setpoint_a == 10.0
        assert s.measured_current_a == 5.0
        assert s.measured_voltage_v == 10.0
        assert s.measured_power_w == pytest.approx(50.0)
        assert s.measured_temp_c == 30.0
        assert s.commanded_output_percent == pytest.approx(10.0)

    def test_status_reports_trips(self) -> None:
        c = _running_ctrl(10.0)
        c.tick(measured_current_a=200.0, measured_voltage_v=20.0, measured_temp_c=25.0)
        s = c.status()
        assert s.state == PowerSupplyState.TRIPPED
        assert "HH_POWER" in s.trips
        assert s.commanded_output_percent == 0.0


# --------------------------------------------------------------------------
# Sanity: math invariants
# --------------------------------------------------------------------------


class TestMathInvariants:
    @pytest.mark.parametrize("sp_a", [0.0, 10.0, 25.0, 50.0])
    def test_command_proportional_to_setpoint(self, sp_a: float) -> None:
        c = _armed_ctrl()
        c.set_current_setpoint(sp_a)
        cmd = c.tick(0.0, 0.0, 25.0)
        expected_pct = 100.0 * sp_a / c.config.current_full_scale_a
        assert cmd == pytest.approx(expected_pct)

    def test_status_power_equals_v_times_i(self) -> None:
        c = _running_ctrl(10.0)
        c.tick(measured_current_a=7.5, measured_voltage_v=12.0, measured_temp_c=25.0)
        s = c.status()
        assert s.measured_power_w == pytest.approx(7.5 * 12.0)
        assert math.isfinite(s.measured_power_w)
