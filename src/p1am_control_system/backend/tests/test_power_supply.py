"""Config-level + state-machine unit tests for PowerSupplyController.

Split companion (`test_power_supply_runtime.py`) covers setpoint clamping,
trip latching, and tick-output behavior under runtime feedback.

Covers here:
    - Pydantic config validation (field constraints, cross-field invariants)
    - Controller construction and initial state
    - Permissive transitions (IDLE <-> ARMED, RUNNING -> IDLE, trip-latch survives)
    - Mode switching between CURRENT and POWER
    - Status snapshot fields and the P = V * I invariant
    - update_config() re-clamps existing setpoint without changing state
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _power_supply_helpers import (
    fresh_armed_controller,
    fresh_armed_controller_unclamped,
    fresh_idle_controller,
    fresh_running_controller,
)
from power_supply import (
    PowerSupplyConfig,
    PowerSupplyController,
    PowerSupplyMode,
    PowerSupplyState,
)
from pydantic import ValidationError

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

    def test_output_clamp_default_is_20_percent(self) -> None:
        assert PowerSupplyConfig().output_clamp_percent == 20.0

    def test_output_clamp_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PowerSupplyConfig(output_clamp_percent=0.0)
        with pytest.raises(ValidationError):
            PowerSupplyConfig(output_clamp_percent=-5.0)

    def test_output_clamp_cannot_exceed_100_percent(self) -> None:
        with pytest.raises(ValidationError):
            PowerSupplyConfig(output_clamp_percent=100.1)

    def test_output_clamp_accepts_full_range_bounds(self) -> None:
        assert PowerSupplyConfig(output_clamp_percent=100.0).output_clamp_percent == (
            100.0
        )

    def test_signal_labels_default(self) -> None:
        cfg = PowerSupplyConfig()
        assert cfg.command_label == "Current command"
        assert cfg.current_feedback_label == "Current"
        assert cfg.voltage_feedback_label == "Voltage"
        assert cfg.temp_label == "Temperature"

    def test_signal_label_is_trimmed(self) -> None:
        assert PowerSupplyConfig(command_label="  Loop A  ").command_label == "Loop A"

    def test_blank_signal_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PowerSupplyConfig(command_label="   ")
        with pytest.raises(ValidationError):
            PowerSupplyConfig(current_feedback_label="")

    def test_overlong_signal_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PowerSupplyConfig(voltage_feedback_label="x" * 41)


# --------------------------------------------------------------------------
# Constructor + initial state
# --------------------------------------------------------------------------


class TestController_Init:
    def test_initial_state_is_idle(self) -> None:
        c = fresh_idle_controller()
        assert c.state == PowerSupplyState.IDLE
        assert c.permissive is False
        assert c.trips == []

    def test_rejects_non_config_argument(self) -> None:
        with pytest.raises(TypeError):
            PowerSupplyController("not a config")
        with pytest.raises(TypeError):
            PowerSupplyController({"current_full_scale_a": 10.0})


# --------------------------------------------------------------------------
# Permissive transitions
# --------------------------------------------------------------------------


class TestPermissive:
    def test_idle_to_armed_when_permissive_on(self) -> None:
        c = fresh_idle_controller()
        c.set_permissive(True)
        assert c.state == PowerSupplyState.ARMED
        assert c.permissive is True

    def test_armed_to_idle_when_permissive_off(self) -> None:
        c = fresh_armed_controller()
        c.set_permissive(False)
        assert c.state == PowerSupplyState.IDLE

    def test_running_to_idle_when_permissive_off(self) -> None:
        c = fresh_running_controller(10.0)
        assert c.state == PowerSupplyState.RUNNING
        c.set_permissive(False)
        assert c.state == PowerSupplyState.IDLE

    def test_permissive_off_clears_setpoint(self) -> None:
        c = fresh_running_controller(15.0)
        c.set_permissive(False)
        # Re-arming gives a clean state at 0 setpoint
        c.set_permissive(True)
        assert c.state == PowerSupplyState.ARMED
        cmd = c.tick(0.0, 0.0, 25.0)
        assert cmd == 0.0

    def test_permissive_change_does_not_clear_trip(self) -> None:
        c = fresh_running_controller(10.0)
        # Drive a trip via measured power
        c.tick(measured_current_a=200.0, measured_voltage_v=20.0, measured_temp_c=25.0)
        assert c.state == PowerSupplyState.TRIPPED
        c.set_permissive(False)
        assert c.state == PowerSupplyState.TRIPPED
        c.set_permissive(True)
        assert c.state == PowerSupplyState.TRIPPED

    def test_rejects_non_bool_permissive(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(TypeError):
            c.set_permissive(1)
        with pytest.raises(TypeError):
            c.set_permissive("on")


# --------------------------------------------------------------------------
# Mode switching
# --------------------------------------------------------------------------


class TestModeSwitching:
    def test_current_setpoint_switches_mode_to_current(self) -> None:
        c = fresh_armed_controller()
        c.tick(0.0, 20.0, 25.0)
        c.set_power_setpoint(200.0)
        assert c.mode == PowerSupplyMode.POWER
        c.set_current_setpoint(15.0)
        assert c.mode == PowerSupplyMode.CURRENT

    def test_power_setpoint_switches_mode_to_power(self) -> None:
        c = fresh_armed_controller()
        c.set_current_setpoint(15.0)
        assert c.mode == PowerSupplyMode.CURRENT
        c.tick(0.0, 20.0, 25.0)
        c.set_power_setpoint(200.0)
        assert c.mode == PowerSupplyMode.POWER


# --------------------------------------------------------------------------
# Status snapshot
# --------------------------------------------------------------------------


class TestStatus:
    def test_effective_max_current_reflects_clamp_and_full_scale(self) -> None:
        # 20 % clamp of a 100 A full scale -> 20 A is the real deliverable max,
        # even though the setpoint band may go to 50 A.
        c = PowerSupplyController(
            PowerSupplyConfig(output_clamp_percent=20.0, current_full_scale_a=100.0)
        )
        assert c.status().effective_max_current_a == pytest.approx(20.0)
        c.update_config(
            PowerSupplyConfig(output_clamp_percent=50.0, current_full_scale_a=80.0)
        )
        assert c.status().effective_max_current_a == pytest.approx(40.0)

    def test_status_reports_current_state(self) -> None:
        c = fresh_running_controller(10.0)
        # First tick establishes the slew baseline at t=0; second tick at
        # t=100 s gives the ramp plenty of headroom to settle on target.
        c.tick(
            measured_current_a=5.0,
            measured_voltage_v=10.0,
            measured_temp_c=30.0,
            now=0.0,
        )
        c.tick(
            measured_current_a=5.0,
            measured_voltage_v=10.0,
            measured_temp_c=30.0,
            now=100.0,
        )
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
        c = fresh_running_controller(10.0)
        c.tick(measured_current_a=200.0, measured_voltage_v=20.0, measured_temp_c=25.0)
        s = c.status()
        assert s.state == PowerSupplyState.TRIPPED
        assert "HH_POWER" in s.trips
        assert s.commanded_output_percent == 0.0


# --------------------------------------------------------------------------
# Math invariants
# --------------------------------------------------------------------------


class TestMathInvariants:
    @pytest.mark.parametrize("sp_a", [0.0, 10.0, 25.0, 50.0])
    def test_command_proportional_to_setpoint(self, sp_a: float) -> None:
        # Unclamped: this checks the proportional law across the full setpoint
        # range; the 20 % output clamp is exercised in TestOutputClamp.
        c = fresh_armed_controller_unclamped()
        c.set_current_setpoint(sp_a)
        # Two ticks with a large dt allow the slew limiter to fully settle.
        c.tick(0.0, 0.0, 25.0, now=0.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=100.0)
        expected_pct = 100.0 * sp_a / c.config.current_full_scale_a
        assert cmd == pytest.approx(expected_pct)

    def test_status_power_equals_v_times_i(self) -> None:
        c = fresh_running_controller(10.0)
        c.tick(measured_current_a=7.5, measured_voltage_v=12.0, measured_temp_c=25.0)
        s = c.status()
        assert s.measured_power_w == pytest.approx(7.5 * 12.0)
        assert math.isfinite(s.measured_power_w)


# --------------------------------------------------------------------------
# update_config runtime behavior
# --------------------------------------------------------------------------


class TestUpdateConfig:
    def test_update_config_re_clamps_existing_setpoint(self) -> None:
        c = fresh_armed_controller()
        c.set_current_setpoint(40.0)
        # New config with stricter max → setpoint should re-clamp.
        new_cfg = PowerSupplyConfig(
            current_full_scale_a=100.0,
            current_setpoint_max_a=20.0,
        )
        c.update_config(new_cfg)
        # Two ticks with large dt so slew settles on the new (clamped) target.
        c.tick(0.0, 0.0, 25.0, now=0.0)
        cmd = c.tick(0.0, 0.0, 25.0, now=100.0)
        # Setpoint should now be 20 A → 20 % of full scale (100 A)
        assert cmd == pytest.approx(20.0)

    def test_update_config_type_check(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(TypeError):
            c.update_config({"current_full_scale_a": 50.0})

    def test_invalid_config_rejected_at_construction(self) -> None:
        # update_config can't take an invalid config because invalid
        # configs cannot be constructed in the first place.
        with pytest.raises(ValidationError):
            PowerSupplyConfig(
                current_full_scale_a=10.0,
                current_setpoint_max_a=50.0,
            )
