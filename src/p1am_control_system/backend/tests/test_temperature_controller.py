"""State-machine + runtime tests for TemperatureController.

Companion (`test_temperature_models.py`) covers config-level validation.

Covers here:
    - Controller construction and initial state
    - Permissive transitions (IDLE <-> ARMED, RUNNING -> IDLE, trip-latch
      survives a permissive change)
    - Setpoint clamping + rejection in IDLE / TRIPPED / E-stopped states
    - On/off hysteresis (below band -> ON, above band -> OFF, inside band holds)
    - HH cutoff latches the relay off and survives a subsequent safe tick
    - acknowledge_trip() transitions
    - E-stop forces the relay off and blocks arm / setpoint
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from temperature_controller import (  # noqa: E402  (path setup must run first)
    TemperatureConfig,
    TemperatureController,
    TemperatureState,
)

# --------------------------------------------------------------------------
# Fresh-controller helpers — known states without repeated boilerplate.
# --------------------------------------------------------------------------


def fresh_idle_controller() -> TemperatureController:
    """Return a controller in IDLE state (default config)."""
    return TemperatureController(TemperatureConfig())


def fresh_armed_controller() -> TemperatureController:
    """Return a controller already in ARMED state (permissive on)."""
    c = fresh_idle_controller()
    c.set_permissive(True)
    return c


def fresh_running_controller(sp_c: float = 500.0) -> TemperatureController:
    """Return a controller in RUNNING state with a non-zero setpoint."""
    c = fresh_armed_controller()
    c.set_setpoint_c(sp_c)
    return c


# --------------------------------------------------------------------------
# Constructor + initial state
# --------------------------------------------------------------------------


class TestControllerInit:
    def test_initial_state_is_idle(self) -> None:
        c = fresh_idle_controller()
        assert c.state == TemperatureState.IDLE
        assert c.permissive is False
        assert c.trips == []
        assert c.estopped is False

    def test_rejects_non_config_argument(self) -> None:
        with pytest.raises(TypeError):
            TemperatureController("not a config")
        with pytest.raises(TypeError):
            TemperatureController({"temp_full_scale_c": 1000.0})

    def test_config_property_returns_config(self) -> None:
        cfg = TemperatureConfig()
        c = TemperatureController(cfg)
        assert c.config is cfg


# --------------------------------------------------------------------------
# Permissive transitions
# --------------------------------------------------------------------------


class TestPermissive:
    def test_idle_to_armed_when_permissive_on(self) -> None:
        c = fresh_idle_controller()
        c.set_permissive(True)
        assert c.state == TemperatureState.ARMED
        assert c.permissive is True

    def test_armed_to_idle_when_permissive_off(self) -> None:
        c = fresh_armed_controller()
        c.set_permissive(False)
        assert c.state == TemperatureState.IDLE

    def test_running_to_idle_when_permissive_off(self) -> None:
        c = fresh_running_controller(500.0)
        assert c.state == TemperatureState.RUNNING
        c.set_permissive(False)
        assert c.state == TemperatureState.IDLE

    def test_permissive_off_clears_setpoint_and_forces_relay_off(self) -> None:
        c = fresh_running_controller(500.0)
        # Below the band -> relay would be ON in RUNNING.
        assert c.tick(100.0) is True
        c.set_permissive(False)
        assert c.tick(100.0) is False
        # Re-arming gives a clean state at 0 setpoint.
        c.set_permissive(True)
        assert c.state == TemperatureState.ARMED
        assert c.tick(100.0) is False  # ARMED never fires

    def test_permissive_change_does_not_clear_trip(self) -> None:
        c = fresh_running_controller(500.0)
        c.tick(measured_temp_c=1500.0)  # breaches HH limit (1400)
        assert c.state == TemperatureState.TRIPPED
        c.set_permissive(False)
        assert c.state == TemperatureState.TRIPPED
        c.set_permissive(True)
        assert c.state == TemperatureState.TRIPPED

    def test_rejects_non_bool_permissive(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(TypeError):
            c.set_permissive(1)
        with pytest.raises(TypeError):
            c.set_permissive("on")


# --------------------------------------------------------------------------
# Setpoint clamping + rejection
# --------------------------------------------------------------------------


class TestSetpoint:
    def test_in_band_setpoint_applied_and_runs(self) -> None:
        c = fresh_armed_controller()
        applied = c.set_setpoint_c(500.0)
        assert applied == 500.0
        assert c.state == TemperatureState.RUNNING

    def test_setpoint_above_max_is_clamped(self) -> None:
        c = fresh_armed_controller()
        applied = c.set_setpoint_c(9999.0)
        assert applied == c.config.setpoint_max_c

    def test_setpoint_below_min_is_clamped(self) -> None:
        c = TemperatureController(TemperatureConfig(setpoint_min_c=100.0))
        c.set_permissive(True)
        applied = c.set_setpoint_c(-50.0)
        assert applied == 100.0

    def test_setpoint_zero_keeps_state_armed(self) -> None:
        c = fresh_armed_controller()
        c.set_setpoint_c(0.0)
        assert c.state == TemperatureState.ARMED

    def test_setpoint_ignored_in_idle_state(self) -> None:
        c = fresh_idle_controller()
        applied = c.set_setpoint_c(500.0)
        assert applied == 500.0  # value clamped + returned
        assert c.state == TemperatureState.IDLE  # but not applied
        assert c.tick(100.0) is False

    def test_setpoint_ignored_in_tripped_state(self) -> None:
        c = fresh_running_controller(500.0)
        c.tick(1500.0)  # trip on HH
        assert c.state == TemperatureState.TRIPPED
        c.set_setpoint_c(300.0)
        assert c.tick(100.0) is False
        assert c.state == TemperatureState.TRIPPED

    def test_setpoint_rejected_while_estopped(self) -> None:
        c = fresh_armed_controller()
        c.engage_estop()
        assert c.set_setpoint_c(500.0) == 0.0

    def test_rejects_nan(self) -> None:
        c = fresh_armed_controller()
        with pytest.raises(ValueError, match="finite"):
            c.set_setpoint_c(float("nan"))

    def test_rejects_infinity(self) -> None:
        c = fresh_armed_controller()
        with pytest.raises(ValueError, match="finite"):
            c.set_setpoint_c(float("inf"))
        with pytest.raises(ValueError, match="finite"):
            c.set_setpoint_c(float("-inf"))

    def test_rejects_non_numeric(self) -> None:
        c = fresh_armed_controller()
        with pytest.raises(TypeError):
            c.set_setpoint_c("500")
        with pytest.raises(TypeError):
            c.set_setpoint_c(None)

    def test_rejects_bool(self) -> None:
        # Bool is technically a subclass of int. Explicit catch keeps the
        # caller honest.
        c = fresh_armed_controller()
        with pytest.raises(TypeError):
            c.set_setpoint_c(True)


# --------------------------------------------------------------------------
# On/off hysteresis control law
# --------------------------------------------------------------------------


class TestHysteresis:
    def test_below_band_turns_relay_on(self) -> None:
        # setpoint 500, deadband 5 -> ON at <= 495.
        c = fresh_running_controller(500.0)
        assert c.tick(490.0) is True
        assert c.status().relay_on is True

    def test_above_band_turns_relay_off(self) -> None:
        # OFF at >= 505. Start with relay on, then drive it off.
        c = fresh_running_controller(500.0)
        assert c.tick(490.0) is True
        assert c.tick(510.0) is False

    def test_inside_band_holds_prior_on_state(self) -> None:
        c = fresh_running_controller(500.0)
        assert c.tick(490.0) is True  # below band -> ON
        # 500 is inside [495, 505] -> hold ON.
        assert c.tick(500.0) is True
        assert c.tick(502.0) is True

    def test_inside_band_holds_prior_off_state(self) -> None:
        c = fresh_running_controller(500.0)
        assert c.tick(510.0) is False  # above band -> OFF
        # 500 is inside the band -> hold OFF.
        assert c.tick(500.0) is False
        assert c.tick(498.0) is False

    def test_relay_toggles_across_full_cycle(self) -> None:
        c = fresh_running_controller(500.0)
        assert c.tick(490.0) is True  # ON
        assert c.tick(500.0) is True  # hold ON
        assert c.tick(510.0) is False  # OFF
        assert c.tick(500.0) is False  # hold OFF
        assert c.tick(494.0) is True  # ON again

    def test_on_threshold_is_inclusive(self) -> None:
        # measured == setpoint - deadband should turn ON.
        c = fresh_running_controller(500.0)
        assert c.tick(495.0) is True

    def test_off_threshold_is_inclusive(self) -> None:
        # measured == setpoint + deadband should turn OFF.
        c = fresh_running_controller(500.0)
        c.tick(490.0)  # ON first
        assert c.tick(505.0) is False

    def test_non_finite_measured_coerced_to_zero(self) -> None:
        # NaN measured -> 0 C, which is well below the band -> relay ON.
        c = fresh_running_controller(500.0)
        assert c.tick(float("nan")) is True
        assert c.status().measured_temp_c == 0.0


# --------------------------------------------------------------------------
# HH cutoff
# --------------------------------------------------------------------------


class TestHHCutoff:
    def test_hh_trips_at_limit(self) -> None:
        c = TemperatureController(TemperatureConfig(hh_limit_c=1000.0))
        c.set_permissive(True)
        c.set_setpoint_c(800.0)
        # measured == hh_limit -> trips (>= is inclusive).
        relay = c.tick(1000.0)
        assert relay is False
        assert c.state == TemperatureState.TRIPPED
        assert "HH_TEMP" in c.trips

    def test_hh_does_not_trip_below_limit(self) -> None:
        c = TemperatureController(TemperatureConfig(hh_limit_c=1000.0))
        c.set_permissive(True)
        c.set_setpoint_c(800.0)
        c.tick(999.0)
        assert c.state == TemperatureState.RUNNING
        assert c.trips == []

    def test_hh_forces_relay_off_even_when_below_band(self) -> None:
        # Without the HH trip, a temp well below the band would turn the relay
        # ON. The HH latch must win and force it off.
        c = TemperatureController(TemperatureConfig(hh_limit_c=1000.0))
        c.set_permissive(True)
        c.set_setpoint_c(800.0)
        c.tick(1000.0)  # latch the trip
        assert c.tick(100.0) is False  # safe/cold tick, but trip latched

    def test_hh_trip_latches_through_safe_subsequent_tick(self) -> None:
        c = TemperatureController(TemperatureConfig(hh_limit_c=1000.0))
        c.set_permissive(True)
        c.set_setpoint_c(800.0)
        c.tick(1000.0)
        assert c.state == TemperatureState.TRIPPED
        # Temperature returns to a safe band; the trip must stay latched.
        c.tick(500.0)
        assert c.state == TemperatureState.TRIPPED
        assert "HH_TEMP" in c.trips


# --------------------------------------------------------------------------
# acknowledge_trip transitions
# --------------------------------------------------------------------------


class TestAcknowledge:
    def test_acknowledge_clears_trips_and_returns_to_armed(self) -> None:
        c = fresh_running_controller(500.0)
        c.tick(1500.0)
        assert c.state == TemperatureState.TRIPPED
        cleared = c.acknowledge_trip()
        assert cleared is True
        assert c.trips == []
        # Permissive was on; ack returns to ARMED, not IDLE.
        assert c.state == TemperatureState.ARMED

    def test_acknowledge_with_permissive_off_returns_to_idle(self) -> None:
        c = fresh_running_controller(500.0)
        c.tick(1500.0)
        c.set_permissive(False)
        c.acknowledge_trip()
        assert c.state == TemperatureState.IDLE

    def test_acknowledge_with_no_trip_returns_false(self) -> None:
        c = fresh_armed_controller()
        assert c.acknowledge_trip() is False


# --------------------------------------------------------------------------
# tick() output guarantees in non-running states
# --------------------------------------------------------------------------


class TestTickOutput:
    def test_tick_returns_false_in_idle(self) -> None:
        c = fresh_idle_controller()
        assert c.tick(100.0) is False

    def test_tick_returns_false_in_armed(self) -> None:
        c = fresh_armed_controller()
        assert c.tick(100.0) is False

    def test_now_argument_is_accepted_and_ignored(self) -> None:
        c = fresh_running_controller(500.0)
        assert c.tick(490.0, now=10.0) is True
        assert c.tick(490.0, now=999.0) is True


# --------------------------------------------------------------------------
# E-stop — latched kill switch
# --------------------------------------------------------------------------


class TestEstop:
    def test_engage_forces_relay_off_from_running(self) -> None:
        c = fresh_running_controller(500.0)
        assert c.tick(100.0) is True  # relay on (below band)
        c.engage_estop()
        assert c.tick(100.0) is False
        assert c.estopped is True
        assert c.state == TemperatureState.IDLE
        assert c.permissive is False

    def test_cannot_arm_while_estopped(self) -> None:
        c = fresh_running_controller(500.0)
        c.engage_estop()
        c.set_permissive(True)  # ignored
        assert c.permissive is False
        assert c.state == TemperatureState.IDLE
        assert c.tick(100.0) is False

    def test_clear_releases_latch_and_requires_rearm(self) -> None:
        c = fresh_running_controller(500.0)
        c.engage_estop()
        c.clear_estop()
        assert c.estopped is False
        assert c.permissive is False  # must re-arm explicitly
        assert c.state == TemperatureState.IDLE
        # After re-arm + setpoint, the relay fires again.
        c.set_permissive(True)
        c.set_setpoint_c(500.0)
        assert c.tick(100.0) is True

    def test_clear_when_not_estopped_is_noop(self) -> None:
        c = fresh_running_controller(500.0)
        c.clear_estop()
        assert c.estopped is False
        assert c.state == TemperatureState.RUNNING


# --------------------------------------------------------------------------
# update_config runtime behavior
# --------------------------------------------------------------------------


class TestUpdateConfig:
    def test_update_config_re_clamps_existing_setpoint(self) -> None:
        c = fresh_running_controller(900.0)
        # New config with a stricter max -> setpoint should re-clamp to 500.
        c.update_config(TemperatureConfig(setpoint_max_c=500.0))
        # State unchanged by the config swap.
        assert c.state == TemperatureState.RUNNING
        # The re-clamped setpoint (500) drives the band: 600 is above
        # 500 + 5 -> relay OFF.
        assert c.tick(600.0) is False
        # 490 is below 500 - 5 -> relay ON, proving the clamp moved the band.
        assert c.tick(490.0) is True

    def test_update_config_type_check(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(TypeError):
            c.update_config({"setpoint_max_c": 500.0})

    def test_update_config_does_not_change_state(self) -> None:
        c = fresh_armed_controller()
        c.update_config(TemperatureConfig(deadband_c=10.0))
        assert c.state == TemperatureState.ARMED
        assert c.config.deadband_c == 10.0
