"""Runtime / feedback-driven setpoint tests for PowerSupplyController.

Split companion: `test_power_supply.py` covers config-level + state-machine
behavior. This file covers setpoint behavior once the controller is fed
measured values through `tick()`:
    - Current setpoint clamping (fat-finger protection) and type/value
      validation (NaN / inf / bool / string rejection)
    - Power setpoint derivation from measured voltage

`test_power_supply_runtime_safety.py` covers trip, output clamp, and slew-rate
behavior.
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
)
from power_supply import PowerSupplyMode, PowerSupplyState

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
        # Unclamped output so this verifies setpoint clamping, not the 20 %
        # output clamp (covered in TestOutputClamp).
        c = fresh_armed_controller_unclamped()
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
        # The setter reports the setpoint now IN EFFECT, not the request. This
        # used to echo 25.0 back, so the HMI showed a command the controller had
        # discarded and the operator went looking for a fault in the load
        # instead of noticing the supply was never armed (issue #4017).
        assert applied == 0.0
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
