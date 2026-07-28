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
    TcPath,
    TcType,
    TemperatureConfig,
    TemperatureController,
    TemperatureState,
    ThermocoupleChannel,
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
# Preload setpoint (boot recall) — seed the held setpoint WITHOUT energizing
# --------------------------------------------------------------------------


class TestPreloadSetpoint:
    """``preload_setpoint_c`` seeds the held setpoint at boot without arming.

    The persisted-setpoint restore path calls this so ``status().setpoint_c``
    reports the recalled target at boot instead of 0 — fixing the "displayed
    setpoint is not the one the controller sees" startup bug — while the
    controller stays IDLE and the relay is force-held off. Applying it only in
    IDLE (and never while E-stopped) means it can never resurrect a target into
    an armed/running/tripped controller.
    """

    def test_seeds_setpoint_in_idle(self) -> None:
        c = fresh_idle_controller()
        applied = c.preload_setpoint_c(700.0)
        assert applied == 700.0
        assert c.status().setpoint_c == 700.0
        assert c.state == TemperatureState.IDLE

    def test_never_energizes(self) -> None:
        # A seeded setpoint in IDLE must NOT turn the relay on: the state
        # machine forces the actuator off in every non-RUNNING state, so even a
        # cold reading (which would call for heat while RUNNING) stays off.
        c = fresh_idle_controller()
        c.preload_setpoint_c(700.0)
        assert c.tick(20.0) is False
        assert c.state == TemperatureState.IDLE
        assert c.status().relay_on is False

    def test_clamps_to_max(self) -> None:
        c = fresh_idle_controller()
        applied = c.preload_setpoint_c(99999.0)
        assert applied == c.config.setpoint_max_c
        assert c.status().setpoint_c == c.config.setpoint_max_c

    def test_clamps_to_min(self) -> None:
        c = TemperatureController(TemperatureConfig(setpoint_min_c=100.0))
        applied = c.preload_setpoint_c(-50.0)
        assert applied == 100.0

    def test_refused_when_armed_leaves_setpoint_zero(self) -> None:
        c = fresh_armed_controller()  # ARMED, setpoint still 0
        assert c.preload_setpoint_c(300.0) == 0.0
        assert c.status().setpoint_c == 0.0  # unchanged

    def test_refused_when_running_preserves_setpoint(self) -> None:
        c = fresh_running_controller(500.0)
        assert c.preload_setpoint_c(300.0) == 0.0
        assert c.status().setpoint_c == 500.0  # running setpoint untouched

    def test_refused_when_tripped(self) -> None:
        c = fresh_running_controller(500.0)
        c.tick(1500.0)  # HH trip
        assert c.state == TemperatureState.TRIPPED
        assert c.preload_setpoint_c(300.0) == 0.0

    def test_refused_when_estopped(self) -> None:
        c = fresh_idle_controller()
        c.engage_estop()
        assert c.preload_setpoint_c(300.0) == 0.0
        assert c.status().setpoint_c == 0.0

    def test_rejects_nan(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(ValueError, match="finite"):
            c.preload_setpoint_c(float("nan"))

    def test_rejects_infinity(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(ValueError, match="finite"):
            c.preload_setpoint_c(float("inf"))
        with pytest.raises(ValueError, match="finite"):
            c.preload_setpoint_c(float("-inf"))

    def test_rejects_non_numeric(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(TypeError):
            c.preload_setpoint_c("700")
        with pytest.raises(TypeError):
            c.preload_setpoint_c(None)

    def test_rejects_bool(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(TypeError):
            c.preload_setpoint_c(True)


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

    def test_non_finite_measured_trips_and_coerces_display(self) -> None:
        # A non-finite reading while RUNNING is a sensor fault: it TRIPS (relay
        # off) rather than coercing to 0 C and calling for heat (see
        # TestSensorFaultTrip). The stored/displayed temperature is still coerced
        # to a finite 0.0 so no NaN leaks into the status/telemetry.
        c = fresh_running_controller(500.0)
        assert c.tick(float("nan")) is False
        assert c.state == TemperatureState.TRIPPED
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
# HH on EITHER thermocouple (dead-controlling-sensor backstop)
# --------------------------------------------------------------------------


class TestHHOnEitherThermocouple:
    """The HH cutoff must trip on the OTHER TC too, so a stuck/dead controlling
    sensor reading a false 'cold' cannot mask a real over-temperature."""

    def test_hh_trips_on_other_tc_while_controlling_tc_reads_cold(self) -> None:
        # Controlling TC stuck cold (30 C) but the other TC is at the HH limit:
        # the vessel is genuinely over-temp and must trip even though the sensor
        # we steer on says it is cold — the exact type-R runaway signature.
        c = TemperatureController(TemperatureConfig(hh_limit_c=1000.0))
        c.set_permissive(True)
        c.set_setpoint_c(800.0)
        relay = c.tick(30.0, other_temp_c=1000.0)
        assert relay is False
        assert c.state == TemperatureState.TRIPPED
        assert "HH_TEMP" in c.trips

    def test_other_tc_below_limit_does_not_trip_hh(self) -> None:
        c = TemperatureController(TemperatureConfig(hh_limit_c=1000.0))
        c.set_permissive(True)
        c.set_setpoint_c(800.0)
        c.tick(700.0, other_temp_c=999.0)
        assert c.state == TemperatureState.RUNNING
        assert c.trips == []

    def test_missing_other_tc_preserves_single_tc_hh(self) -> None:
        # other_temp_c=None (single-TC callers / bench with one probe): HH still
        # evaluates on the controlling TC exactly as before.
        c = TemperatureController(TemperatureConfig(hh_limit_c=1000.0))
        c.set_permissive(True)
        c.set_setpoint_c(800.0)
        assert c.tick(1000.0) is False
        assert c.state == TemperatureState.TRIPPED
        assert "HH_TEMP" in c.trips


# --------------------------------------------------------------------------
# Cross-check trip: controlling sensor stuck cold while the other reads hot
# --------------------------------------------------------------------------


class TestCrossCheckTrip:
    """Debounced TC_DISAGREE trip — the fast catch for a dead controlling
    thermocouple driving a runaway before HH (the type-R incident)."""

    def _run_disagreeing_scans(
        self, c: TemperatureController, n: int, *, active: float = 30.0
    ) -> None:
        for _ in range(n):
            c.tick(active, other_temp_c=800.0)

    def test_trips_after_debounce_when_controlling_tc_stuck_cold(self) -> None:
        from temperature_controller import _CROSS_FAULT_DEBOUNCE_SCANS

        c = fresh_running_controller(500.0)
        # One scan short of the debounce: still RUNNING, no trip yet.
        self._run_disagreeing_scans(c, _CROSS_FAULT_DEBOUNCE_SCANS - 1)
        assert c.state == TemperatureState.RUNNING
        assert "TC_DISAGREE" not in c.trips
        # The scan that reaches the debounce latches the trip and kills the relay.
        relay = c.tick(30.0, other_temp_c=800.0)
        assert relay is False
        assert c.state == TemperatureState.TRIPPED
        assert "TC_DISAGREE" in c.trips

    def test_debounce_resets_on_a_good_scan(self) -> None:
        from temperature_controller import _CROSS_FAULT_DEBOUNCE_SCANS

        c = fresh_running_controller(500.0)
        self._run_disagreeing_scans(c, _CROSS_FAULT_DEBOUNCE_SCANS - 1)
        # A single agreeing scan (both plausibly warm) breaks the streak...
        c.tick(400.0, other_temp_c=420.0)
        assert "TC_DISAGREE" not in c.trips
        # ...so it now takes a full fresh debounce window to trip again.
        self._run_disagreeing_scans(c, _CROSS_FAULT_DEBOUNCE_SCANS - 1)
        assert c.state == TemperatureState.RUNNING
        assert "TC_DISAGREE" not in c.trips

    def test_no_trip_at_startup_when_both_cold(self) -> None:
        # Both TCs near ambient during warm-up must never cross-trip.
        c = fresh_running_controller(500.0)
        for _ in range(20):
            c.tick(25.0, other_temp_c=27.0)
        assert c.state == TemperatureState.RUNNING
        assert "TC_DISAGREE" not in c.trips

    def test_no_trip_when_not_running(self) -> None:
        # ARMED (not yet RUNNING): the control law can't energize, so a cold/hot
        # split is not a runaway and must not trip.
        c = fresh_armed_controller()
        for _ in range(20):
            c.tick(30.0, other_temp_c=800.0)
        assert c.state == TemperatureState.ARMED
        assert "TC_DISAGREE" not in c.trips

    def test_no_trip_when_controlling_tc_is_the_hot_one(self) -> None:
        # The GOOD sensor is controlling (hot) and the OTHER is the broken-cold
        # one: we are not steering on a dead sensor, so no cross-trip.
        c = fresh_running_controller(500.0)
        for _ in range(20):
            c.tick(480.0, other_temp_c=30.0)
        assert c.state == TemperatureState.RUNNING
        assert "TC_DISAGREE" not in c.trips

    def test_missing_other_tc_never_cross_trips(self) -> None:
        c = fresh_running_controller(500.0)
        for _ in range(20):
            c.tick(30.0)  # no other_temp_c -> no cross-check data
        assert c.state == TemperatureState.RUNNING
        assert "TC_DISAGREE" not in c.trips


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


# --------------------------------------------------------------------------
# Anti-short-cycle min on/off dwell
# --------------------------------------------------------------------------


def running_with_dwell(
    min_on: float = 0.0, min_off: float = 0.0, sp_c: float = 500.0
) -> TemperatureController:
    """A RUNNING controller (setpoint 500, deadband 5) with dwell limits set."""
    c = TemperatureController(
        TemperatureConfig(min_on_time_s=min_on, min_off_time_s=min_off)
    )
    c.set_permissive(True)
    c.set_setpoint_c(sp_c)
    return c


class TestAntiShortCycle:
    def test_min_off_time_blocks_early_re_energize(self) -> None:
        c = running_with_dwell(min_off=30.0)
        assert c.tick(490.0, now=0.0) is True  # first ON allowed (no prior switch)
        assert c.tick(510.0, now=1.0) is False  # OFF (min_on=0)
        # wants ON again at t=10, but only 9 s off < 30 s -> held OFF
        assert c.tick(490.0, now=10.0) is False
        # 30 s elapsed -> re-energize allowed
        assert c.tick(490.0, now=31.0) is True

    def test_min_on_time_blocks_early_de_energize(self) -> None:
        c = running_with_dwell(min_on=30.0)
        assert c.tick(490.0, now=0.0) is True  # ON
        # wants OFF at t=10, but only 10 s on < 30 s -> held ON
        assert c.tick(510.0, now=10.0) is True
        # 30 s elapsed -> de-energize allowed
        assert c.tick(510.0, now=31.0) is False

    def test_dwell_boundary_is_inclusive(self) -> None:
        c = running_with_dwell(min_off=30.0)
        assert c.tick(490.0, now=0.0) is True
        assert c.tick(510.0, now=1.0) is False  # OFF at t=1
        # exactly 30 s after the OFF switch -> allowed (>= not >)
        assert c.tick(490.0, now=31.0) is True

    def test_estop_bypasses_dwell(self) -> None:
        c = running_with_dwell(min_on=60.0)
        assert c.tick(490.0, now=0.0) is True  # ON
        c.engage_estop()
        # min_on not elapsed, but E-stop forces OFF the same tick
        assert c.tick(490.0, now=1.0) is False

    def test_permissive_off_bypasses_dwell(self) -> None:
        c = running_with_dwell(min_on=60.0)
        assert c.tick(490.0, now=0.0) is True
        c.set_permissive(False)
        assert c.tick(490.0, now=1.0) is False

    def test_no_clock_disables_dwell(self) -> None:
        c = running_with_dwell(min_off=60.0)
        assert c.tick(490.0) is True  # ON (now=None)
        assert c.tick(510.0) is False  # OFF
        assert c.tick(490.0) is True  # immediately ON again — dwell not enforced

    def test_default_config_has_no_dwell(self) -> None:
        c = fresh_running_controller()
        assert c.tick(490.0, now=0.0) is True
        assert c.tick(510.0, now=0.1) is False
        assert c.tick(490.0, now=0.2) is True  # no min-off constraint by default

    def test_status_reports_dwell_config(self) -> None:
        c = running_with_dwell(min_on=15.0, min_off=20.0)
        st = c.status()
        assert st.min_on_time_s == 15.0
        assert st.min_off_time_s == 20.0


# --------------------------------------------------------------------------
# Active thermocouple selection (type K / type R toggle)
# --------------------------------------------------------------------------


class TestActiveTcType:
    def test_default_active_is_type_k(self) -> None:
        c = fresh_idle_controller()
        assert c.config.active_tc_type == TcType.TYPE_K
        assert c.config.temp_tag == "TAG_0"

    def test_switch_to_type_r_changes_active_tag(self) -> None:
        c = fresh_idle_controller()
        c.set_active_tc_type(TcType.TYPE_R)
        assert c.config.active_tc_type == TcType.TYPE_R
        assert c.config.temp_tag == "TAG_1"  # the type-R channel's tag
        assert c.status().active_tc_type == TcType.TYPE_R

    def test_switch_back_to_type_k(self) -> None:
        c = fresh_idle_controller()
        c.set_active_tc_type(TcType.TYPE_R)
        c.set_active_tc_type(TcType.TYPE_K)
        assert c.config.active_tc_type == TcType.TYPE_K
        assert c.config.temp_tag == "TAG_0"

    def test_switch_does_not_change_state_machine(self) -> None:
        c = fresh_running_controller(sp_c=500.0)
        assert c.state == TemperatureState.RUNNING
        c.set_active_tc_type(TcType.TYPE_R)
        assert c.state == TemperatureState.RUNNING

    def test_rejects_non_tc_type(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(TypeError):
            c.set_active_tc_type("R")  # type: ignore[arg-type]

    def test_switch_reclamps_setpoint_to_narrower_channel(self) -> None:
        # type-R channel has a smaller full scale; switching must pull the
        # setpoint / limits down so the resulting config stays valid.
        cfg = TemperatureConfig(
            type_r=ThermocoupleChannel(tag="TAG_1", full_scale_c=900.0, label="R"),
        )
        c = TemperatureController(cfg)
        c.set_permissive(True)
        c.set_setpoint_c(1200.0)  # valid under active K (1400)
        c.set_active_tc_type(TcType.TYPE_R)
        assert c.config.setpoint_max_c <= 900.0
        assert c.status().setpoint_c <= 900.0

    def test_active_channel_drives_status_label(self) -> None:
        cfg = TemperatureConfig(
            type_r=ThermocoupleChannel(
                tag="TAG_1", full_scale_c=1400.0, label="R-furnace"
            ),
        )
        c = TemperatureController(cfg)
        c.set_active_tc_type(TcType.TYPE_R)
        assert c.status().active_tc_label == "R-furnace"

    def test_set_active_tc_type_preserves_the_current_path(self) -> None:
        # The backward-compat type-only shim must not silently reset the path.
        c = fresh_idle_controller()
        c.set_active_source(TcType.TYPE_K, TcPath.ANALOG)
        c.set_active_tc_type(TcType.TYPE_R)
        assert c.config.active_tc_type == TcType.TYPE_R
        assert c.config.active_tc_path == TcPath.ANALOG  # path untouched


class TestActiveSource:
    """The 2x2 source selector (type x path) over set_active_source."""

    def test_default_source_is_type_k_on_the_tc_card(self) -> None:
        c = fresh_idle_controller()
        assert c.config.active_tc_type == TcType.TYPE_K
        assert c.config.active_tc_path == TcPath.TC_CARD
        assert c.config.temp_tag == "TAG_0"

    def test_switch_to_analog_r_changes_tag_and_status(self) -> None:
        c = fresh_idle_controller()
        c.set_active_source(TcType.TYPE_R, TcPath.ANALOG)
        assert c.config.active_tc_type == TcType.TYPE_R
        assert c.config.active_tc_path == TcPath.ANALOG
        assert c.config.temp_tag == "TAG_15"  # analog R -> AI3
        st = c.status()
        assert st.active_tc_path == TcPath.ANALOG
        assert st.active_tc_label == "Type R (Analog)"

    def test_each_of_the_four_sources_selects_its_tag(self) -> None:
        c = fresh_idle_controller()
        expected = {
            (TcType.TYPE_K, TcPath.TC_CARD): "TAG_0",
            (TcType.TYPE_R, TcPath.TC_CARD): "TAG_1",
            (TcType.TYPE_K, TcPath.ANALOG): "TAG_14",
            (TcType.TYPE_R, TcPath.ANALOG): "TAG_15",
        }
        for (tc_type, tc_path), tag in expected.items():
            c.set_active_source(tc_type, tc_path)
            assert c.config.temp_tag == tag

    def test_switch_does_not_change_state_machine(self) -> None:
        c = fresh_running_controller(sp_c=500.0)
        c.set_active_source(TcType.TYPE_K, TcPath.ANALOG)
        assert c.state == TemperatureState.RUNNING

    def test_reclamps_setpoint_to_the_new_analog_full_scale(self) -> None:
        cfg = TemperatureConfig(
            analog_k=ThermocoupleChannel(
                tag="TAG_14", full_scale_c=900.0, label="Type K (Analog)"
            ),
        )
        c = TemperatureController(cfg)
        c.set_permissive(True)
        c.set_setpoint_c(1200.0)  # valid under active TC-card K (1400)
        c.set_active_source(TcType.TYPE_K, TcPath.ANALOG)
        assert c.config.setpoint_max_c <= 900.0
        assert c.status().setpoint_c <= 900.0

    def test_rejects_non_tc_type(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(TypeError):
            c.set_active_source("K", TcPath.ANALOG)  # type: ignore[arg-type]

    def test_rejects_non_tc_path(self) -> None:
        c = fresh_idle_controller()
        with pytest.raises(TypeError):
            c.set_active_source(TcType.TYPE_K, "analog")  # type: ignore[arg-type]

    def test_ignored_while_estopped(self) -> None:
        c = fresh_idle_controller()
        c.engage_estop()
        c.set_active_source(TcType.TYPE_R, TcPath.ANALOG)
        assert c.config.active_tc_type == TcType.TYPE_K  # unchanged
        assert c.config.active_tc_path == TcPath.TC_CARD


class TestReadingCoercion:
    def test_non_numeric_reading_coerced_to_zero(self) -> None:
        # IDLE forces the relay off, but _safe_finite still runs on the input:
        # a None / non-numeric reading must coerce to 0 (deterministic), not crash.
        c = fresh_idle_controller()
        assert c.tick(None) is False  # type: ignore[arg-type]
        assert c.tick("sensor fault") is False  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Regression: safety hardening (open-TC runaway trip + E-stop config lock)
# --------------------------------------------------------------------------
class TestSensorFaultTrip:
    """A non-finite feedback while RUNNING must trip, not read 'cold' and heat."""

    def test_running_nan_reading_latches_tc_fault_and_kills_relay(self) -> None:
        c = fresh_running_controller(500.0)
        assert c.state == TemperatureState.RUNNING
        # Baseline: a genuine cold reading calls for heat.
        assert c.tick(20.0) is True
        # Open thermocouple (NaN) — must trip instead of coercing to 0 and heating.
        assert c.tick(float("nan")) is False
        assert c.state == TemperatureState.TRIPPED
        assert "TC_FAULT" in c.trips

    def test_running_inf_reading_trips(self) -> None:
        c = fresh_running_controller(500.0)
        assert c.tick(float("inf")) is False
        assert c.state == TemperatureState.TRIPPED
        assert "TC_FAULT" in c.trips

    def test_idle_nan_reading_does_not_spuriously_trip(self) -> None:
        # In non-energizable states the relay is already forced off, so a junk
        # reading during bring-up must NOT latch a spurious trip.
        c = fresh_idle_controller()
        assert c.tick(float("nan")) is False
        assert c.state == TemperatureState.IDLE
        assert c.trips == []

    def test_finite_reading_never_trips_tc_fault(self) -> None:
        c = fresh_running_controller(500.0)
        for t in (0.0, 25.0, 300.0):
            c.tick(t)
        assert "TC_FAULT" not in c.trips


class TestEstopConfigLock:
    """A latched E-stop must reject config / TC-type mutation (one-way kill)."""

    def test_update_config_ignored_while_estopped(self) -> None:
        c = fresh_running_controller(500.0)
        original_hh = c.config.hh_limit_c
        c.engage_estop()
        c.update_config(TemperatureConfig(hh_limit_c=original_hh - 100.0))
        assert c.config.hh_limit_c == original_hh  # unchanged

    def test_set_active_tc_type_ignored_while_estopped(self) -> None:
        c = fresh_running_controller(500.0)
        original = c.config.active_tc_type
        c.engage_estop()
        other = TcType.TYPE_R if original == TcType.TYPE_K else TcType.TYPE_K
        c.set_active_tc_type(other)
        assert c.config.active_tc_type == original  # unchanged

    def test_type_errors_still_raise_even_when_estopped(self) -> None:
        # DbC precondition checks precede the E-stop no-op guard.
        c = fresh_running_controller(500.0)
        c.engage_estop()
        with pytest.raises(TypeError):
            c.update_config("not a config")
        with pytest.raises(TypeError):
            c.set_active_tc_type("K")
