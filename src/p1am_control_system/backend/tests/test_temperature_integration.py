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
from temperature_models import TemperatureConfig, TemperatureState  # noqa: E402


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
