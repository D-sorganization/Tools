"""Tests for the TemperatureService and FastAPI router wiring.

Covers:
    - TemperatureService.poll() scales the thermocouple tag (percent of full
      scale) into deg C, ticks the controller, and drives the heater relay
      coil via the client's public write_coil seam.
    - The relay is commanded OFF whenever the controller forces it off
      (not running / permissive off / E-stop / HH trip), and a failed coil
      write never aborts the scan.
    - The FastAPI router exposes every documented endpoint with the right
      shape (GET/PUT /config, GET /status, POST /setpoint, POST /permissive,
      POST /acknowledge_trip).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import hardware  # noqa: E402
from temperature_integration import (  # noqa: E402
    TemperatureService,
    create_temperature_router,
)
from temperature_models import (  # noqa: E402
    TcType,
    TemperatureConfig,
    TemperatureState,
)


class _FakePLC:
    """Minimal PLC double — only the coil seam TemperatureService touches."""

    def __init__(self, *, connected: bool = True) -> None:
        self.connected = connected
        self.write_coil = AsyncMock(return_value=True)


def _service(plc: Any | None = None) -> TemperatureService:
    return TemperatureService(plc or _FakePLC(), logging.getLogger("test"))


# --------------------------------------------------------------------------
# poll + tag scaling
# --------------------------------------------------------------------------


class TestTemperatureServicePoll:
    def test_poll_with_no_tags_is_zero_and_relay_off(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = _service(plc)
            status = await svc.poll(None)
            assert status.measured_temp_c == 0.0
            assert status.relay_on is False
            assert status.state == TemperatureState.IDLE
            plc.write_coil.assert_awaited_with(hardware.HEATER_RELAY_COIL, False)

        asyncio.run(_go())

    def test_temp_from_tags_scales_percent_to_celsius(self) -> None:
        svc = _service()
        cfg = TemperatureConfig(temp_full_scale_c=1400.0, temp_tag="TAG_0")
        svc.controller.update_config(cfg)
        # 50 % of a 1400 C range -> 700 C
        assert svc._temp_from_tags({"TAG_0": 50.0}) == pytest.approx(700.0)
        assert svc._temp_from_tags({}) == 0.0
        assert svc._temp_from_tags(None) == 0.0

    def test_poll_closes_relay_when_running_and_below_band(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = _service(plc)
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(700.0)  # ARMED -> RUNNING
            # Measured well below (setpoint - deadband) -> heater should close.
            status = await svc.poll({"TAG_0": 0.0})
            assert status.relay_on is True
            plc.write_coil.assert_awaited_with(hardware.HEATER_RELAY_COIL, True)

        asyncio.run(_go())

    def test_poll_opens_relay_above_band(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = _service(plc)
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(700.0)
            # 60 % of 1400 = 840 C, above setpoint + deadband -> relay off.
            status = await svc.poll({"TAG_0": 60.0})
            assert status.relay_on is False
            plc.write_coil.assert_awaited_with(hardware.HEATER_RELAY_COIL, False)

        asyncio.run(_go())

    def test_hh_cutoff_latches_relay_off(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = _service(plc)
            svc.controller.update_config(
                TemperatureConfig(hh_limit_c=1000.0, temp_full_scale_c=1400.0)
            )
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(700.0)
            # 80 % of 1400 = 1120 C >= hh_limit -> trip + relay off.
            status = await svc.poll({"TAG_0": 80.0})
            assert status.relay_on is False
            assert status.state == TemperatureState.TRIPPED
            assert "HH_TEMP" in status.trips

        asyncio.run(_go())

    def test_estop_forces_relay_off(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = _service(plc)
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(700.0)
            svc.engage_estop()
            status = await svc.poll({"TAG_0": 0.0})
            assert status.relay_on is False
            assert status.estopped is True
            plc.write_coil.assert_awaited_with(hardware.HEATER_RELAY_COIL, False)

        asyncio.run(_go())

    def test_failed_coil_write_does_not_raise(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            plc.write_coil = AsyncMock(side_effect=RuntimeError("bus dropped"))
            svc = _service(plc)
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(700.0)
            # Must not propagate — poll returns a status regardless.
            status = await svc.poll({"TAG_0": 0.0})
            assert status.state == TemperatureState.RUNNING

        asyncio.run(_go())


# --------------------------------------------------------------------------
# FastAPI router endpoints
# --------------------------------------------------------------------------


@pytest.fixture
def client_and_service(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, TemperatureService]:
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    svc = _service()
    app = FastAPI()
    app.include_router(create_temperature_router(svc))
    return TestClient(app), svc


class TestRouterEndpoints:
    def test_get_config_returns_defaults(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, _ = client_and_service
        resp = client.get("/api/temperature/config")
        assert resp.status_code == 200
        body = resp.json()
        assert body["temp_full_scale_c"] == 1400.0
        assert body["hh_limit_c"] == 1400.0

    def test_put_config_validates_and_replaces(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, svc = client_and_service
        new_cfg = svc.controller.config.model_dump()
        new_cfg["deadband_c"] = 10.0
        resp = client.put("/api/temperature/config", json=new_cfg)
        assert resp.status_code == 200
        assert svc.controller.config.deadband_c == 10.0

    def test_put_config_invalid_invariant_returns_422(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, svc = client_and_service
        bad = svc.controller.config.model_dump()
        bad["hh_limit_c"] = 9999.0  # exceeds temp_full_scale_c
        resp = client.put("/api/temperature/config", json=bad)
        assert resp.status_code == 422

    def test_permissive_toggle_arms(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, _ = client_and_service
        resp = client.post("/api/temperature/permissive", json={"enabled": True})
        assert resp.json()["state"] == TemperatureState.ARMED

    def test_setpoint_applies_and_clamps(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, _ = client_and_service
        client.post("/api/temperature/permissive", json={"enabled": True})
        resp = client.post("/api/temperature/setpoint", json={"value_c": 99999.0})
        assert resp.status_code == 200
        assert resp.json()["applied_c"] == 1400.0  # clamped to setpoint_max_c

    def test_acknowledge_trip_no_op_returns_status(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, _ = client_and_service
        resp = client.post("/api/temperature/acknowledge_trip")
        assert resp.status_code == 200
        assert resp.json()["state"] == TemperatureState.IDLE


# --------------------------------------------------------------------------
# Active thermocouple (type K / type R) selection
# --------------------------------------------------------------------------


class TestActiveThermocouple:
    def test_poll_follows_active_channel(self) -> None:
        async def _go() -> None:
            svc = _service()
            # Type K active -> controller reads TAG_0.
            status = await svc.poll({"TAG_0": 50.0, "TAG_1": 0.0})
            assert status.measured_temp_c == pytest.approx(700.0)
            assert status.active_tc_type == TcType.TYPE_K
            # Switch to type R -> controller reads TAG_1 instead.
            svc.controller.set_active_tc_type(TcType.TYPE_R)
            status = await svc.poll({"TAG_0": 0.0, "TAG_1": 50.0})
            assert status.measured_temp_c == pytest.approx(700.0)
            assert status.active_tc_type == TcType.TYPE_R

        asyncio.run(_go())

    def test_tc_type_endpoint_switches_active(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, svc = client_and_service
        resp = client.post("/api/temperature/tc_type", json={"active_tc_type": "R"})
        assert resp.status_code == 200
        assert resp.json()["active_tc_type"] == "R"
        assert svc.controller.config.active_tc_type == TcType.TYPE_R

    def test_tc_type_endpoint_rejects_unknown_type(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, _ = client_and_service
        resp = client.post("/api/temperature/tc_type", json={"active_tc_type": "X"})
        assert resp.status_code == 422

    def test_status_reports_active_tc_type(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, _ = client_and_service
        body = client.get("/api/temperature/status").json()
        assert body["active_tc_type"] == "K"
        assert "active_tc_label" in body


class TestBothThermocoupleReadings:
    """The service surfaces BOTH TC readings every scan and feeds the
    non-controlling one to the controller as the cross-check reference."""

    def test_status_reports_both_tc_temps_regardless_of_active(self) -> None:
        async def _go() -> None:
            svc = _service()  # type K active by default (TAG_0)
            # K=50% -> 700 C, R=20% -> 280 C. Both reported even though K controls.
            status = await svc.poll({"TAG_0": 50.0, "TAG_1": 20.0})
            assert status.type_k_temp_c == pytest.approx(700.0)
            assert status.type_r_temp_c == pytest.approx(280.0)
            assert status.measured_temp_c == pytest.approx(700.0)  # active == K

        asyncio.run(_go())

    def test_both_tc_temps_none_before_first_scan(self) -> None:
        svc = _service()
        status = svc.status()
        assert status.type_k_temp_c is None
        assert status.type_r_temp_c is None

    def test_no_tags_leaves_both_tc_temps_none(self) -> None:
        async def _go() -> None:
            svc = _service()
            status = await svc.poll(None)
            assert status.type_k_temp_c is None
            assert status.type_r_temp_c is None

        asyncio.run(_go())

    def test_cross_check_trips_when_controlling_tc_stuck_cold(self) -> None:
        # The type-R incident end-to-end through the service: control off K but
        # K is stuck cold (~34 C) while R (the other TC) is genuinely hot. After
        # the debounce the controller must latch TC_DISAGREE and open the relay.
        async def _go() -> None:
            from temperature_controller import _CROSS_FAULT_DEBOUNCE_SCANS

            plc = _FakePLC()
            svc = _service(plc)
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(700.0)  # RUNNING, K controls
            # K ~34 C (2.4% of 1400) stuck cold; R ~800 C (57.1%) hot.
            tags = {"TAG_0": 2.4, "TAG_1": 57.1}
            status = None
            for _ in range(_CROSS_FAULT_DEBOUNCE_SCANS):
                status = await svc.poll(tags)
            assert status is not None
            assert status.state == TemperatureState.TRIPPED
            assert "TC_DISAGREE" in status.trips
            assert status.relay_on is False

        asyncio.run(_go())

    def test_switching_to_a_dead_cold_tc_while_hot_trips(self) -> None:
        # Operator switches to control off R while R is dead-cold and the vessel
        # (K) is hot: the cross-check catches steering onto a dead sensor.
        async def _go() -> None:
            from temperature_controller import _CROSS_FAULT_DEBOUNCE_SCANS

            svc = _service()
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(700.0)
            svc.set_active_tc_type(TcType.TYPE_R)  # smooth switch, stays RUNNING
            assert svc.controller.state == TemperatureState.RUNNING
            # R stuck cold (2.4% -> ~34 C), K hot (57.1% -> ~800 C).
            status = None
            for _ in range(_CROSS_FAULT_DEBOUNCE_SCANS):
                status = await svc.poll({"TAG_0": 57.1, "TAG_1": 2.4})
            assert status is not None
            assert "TC_DISAGREE" in status.trips

        asyncio.run(_go())


class TestControlSensorDeglitch:
    """The per-channel deglitch filter must stop a burnout-zero on the control
    thermocouple from reaching the control law as a spurious 'cold', hold the
    last-good value through the glitch, and fail safe if the fault persists."""

    # 82.14 % of 1400 C ~= 1150 C.
    HOT_PCT = 82.14
    HOT_C = 82.14 * 1400.0 / 100.0

    def test_burnout_zero_on_control_tc_is_held_not_fed_cold(self) -> None:
        async def _go() -> None:
            svc = _service()
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(1200.0)  # RUNNING, K controls
            warm = await svc.poll({"TAG_0": self.HOT_PCT, "TAG_1": self.HOT_PCT})
            assert warm.measured_temp_c == pytest.approx(self.HOT_C, abs=1)
            # K burns out to 0 for one scan -> held at last-good, NOT 0.
            glitch = await svc.poll({"TAG_0": 0.0, "TAG_1": self.HOT_PCT})
            assert glitch.measured_temp_c == pytest.approx(self.HOT_C, abs=1)
            assert glitch.control_sensor_holding is True
            assert glitch.state == TemperatureState.RUNNING
            assert "TC_FAULT" not in glitch.trips

        asyncio.run(_go())

    def test_holding_clears_on_recovery(self) -> None:
        async def _go() -> None:
            svc = _service()
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(1200.0)
            await svc.poll({"TAG_0": self.HOT_PCT, "TAG_1": self.HOT_PCT})
            held = await svc.poll({"TAG_0": 0.0, "TAG_1": self.HOT_PCT})
            assert held.control_sensor_holding is True
            recovered = await svc.poll({"TAG_0": 82.5, "TAG_1": self.HOT_PCT})
            assert recovered.control_sensor_holding is False
            assert recovered.measured_temp_c == pytest.approx(82.5 * 14, abs=1)

        asyncio.run(_go())

    def test_persistent_control_sensor_fault_trips_heater(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drive the injected monotonic clock past the filter's hold timeout so a
        # sustained burnout escalates to a latched TC_FAULT (fail-safe stop).
        import temperature_integration as ti

        clock = {"t": 0.0}
        monkeypatch.setattr(ti.time, "monotonic", lambda: clock["t"])

        async def _go() -> None:
            svc = _service()
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(1200.0)
            await svc.poll({"TAG_0": self.HOT_PCT, "TAG_1": self.HOT_PCT})
            clock["t"] = 1.0
            still = await svc.poll({"TAG_0": 0.0, "TAG_1": self.HOT_PCT})
            assert still.control_sensor_holding is True
            assert "TC_FAULT" not in still.trips  # brief glitch: still holding
            clock["t"] = 30.0  # well past the 15 s hold timeout
            tripped = await svc.poll({"TAG_0": 0.0, "TAG_1": self.HOT_PCT})
            assert "TC_FAULT" in tripped.trips
            assert tripped.state == TemperatureState.TRIPPED
            assert tripped.relay_on is False

        asyncio.run(_go())

    def test_other_tc_burnout_does_not_poison_cross_check(self) -> None:
        # Control off K (hot). The OTHER TC (R) burning out to 0 must be held, so
        # it never looks like a spurious "cold other" — control keeps running.
        async def _go() -> None:
            svc = _service()
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(1200.0)
            await svc.poll({"TAG_0": self.HOT_PCT, "TAG_1": self.HOT_PCT})
            s = await svc.poll({"TAG_0": self.HOT_PCT, "TAG_1": 0.0})
            assert s.type_r_temp_c == pytest.approx(self.HOT_C, abs=1)  # R held
            assert s.state == TemperatureState.RUNNING
            assert s.trips == []

        asyncio.run(_go())


class TestServiceAndTcTypeErrors:
    def test_service_estop_engage_and_clear(self) -> None:
        svc = _service()
        svc.engage_estop()
        assert svc.controller.estopped is True
        svc.clear_estop()
        assert svc.controller.estopped is False

    def test_tc_type_endpoint_400_on_degenerate_channel(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, svc = client_and_service
        cfg = svc.controller.config.model_dump()
        cfg["setpoint_min_c"] = 50.0
        cfg["type_r"] = {"tag": "TAG_1", "full_scale_c": 10.0, "label": "R"}
        assert client.put("/api/temperature/config", json=cfg).status_code == 200
        # Switching to R clamps the band below setpoint_min -> invalid -> 400.
        resp = client.post("/api/temperature/tc_type", json={"active_tc_type": "R"})
        assert resp.status_code == 400
