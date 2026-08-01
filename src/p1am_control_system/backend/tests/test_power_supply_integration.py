"""Tests for the PowerSupplyService and FastAPI router wiring.

Covers:
    - PowerSupplyService.poll() feeds the controller and applies its command
      to the PLC via the PID-pass-through write.
    - _inputs_from_tags scales raw 0..100 % tag values to engineering units
      using the controller's configured full-scale settings.
    - The PID setpoint write path: disconnect short-circuits to False; valid
      pid_index range; error response is logged and returns False;
      exceptions are caught.
    - The FastAPI router exposes every documented endpoint with the right
      shape (GET/PUT /config, GET /status, POST /setpoint with current and
      power modes, POST /permissive, POST /acknowledge_trip).
    - Setpoint requests with the wrong field for the chosen mode (e.g.
      mode=current but no value_a) return HTTP 400.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from power_supply import PowerSupplyConfig, PowerSupplyState
from power_supply_integration import (
    PowerSupplyService,
    create_power_supply_router,
)

# --------------------------------------------------------------------------
# Fake PLC client — only implements what PowerSupplyService touches.
# --------------------------------------------------------------------------


class _FakePLC:
    def __init__(self, *, connected: bool = True) -> None:
        self.connected = connected
        self.lock = _NullAsyncLock()
        self._raw = MagicMock()
        self._raw.write_registers = AsyncMock(
            return_value=MagicMock(isError=lambda: False)
        )
        # Public command seam the service delegates to.
        self.write_pid_setpoint = AsyncMock(return_value=True)

    def _get_client(self) -> Any:
        return self._raw


class _NullAsyncLock:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: Any) -> None:
        return None


# --------------------------------------------------------------------------
# PowerSupplyService.poll + helpers
# --------------------------------------------------------------------------


class TestPowerSupplyServicePoll:
    def test_poll_with_no_tags_raises_a_sensor_fault(self) -> None:
        """A scan with no feedback is a fault, not three readings of zero.

        Substituting zeros made HH_POWER and HH_TEMP permanently un-trippable
        and reported a confident, cold-looking supply while the output stayed
        energized (issue #4016).
        """

        async def _go() -> None:
            plc = _FakePLC()
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            status = await svc.poll(None)
            assert "SENSOR_FAULT" in status.trips
            assert status.state == PowerSupplyState.TRIPPED

        asyncio.run(_go())

    def test_poll_writes_pid_setpoint_when_running(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            svc.controller.set_permissive(True)
            svc.controller.set_current_setpoint(40.0)
            await svc.poll({})
            await svc.poll({})
            # The service delegates to the client's public seam (PID 0 = the
            # power-supply actuator). It no longer reaches into write_registers.
            assert plc.write_pid_setpoint.await_count >= 1
            call = plc.write_pid_setpoint.await_args
            assert call is not None
            assert call.args[0] == 0

        asyncio.run(_go())

    def test_inputs_from_tags_scales_percent_to_engineering(self) -> None:
        plc = _FakePLC()
        svc = PowerSupplyService(plc, logging.getLogger("test"))
        cfg = PowerSupplyConfig(current_full_scale_a=200.0, voltage_full_scale_v=100.0)
        svc.controller.update_config(cfg)
        # 50 % on current feedback tag → 100 A
        # 25 % on voltage feedback tag → 25 V
        # Temp is a PERCENT of full scale like the other tags, not degC. This
        # assertion previously pinned the pass-through as correct, which is why
        # the HH_TEMP trip could never fire (issue #4003).
        tags = {
            cfg.current_feedback_tag: 50.0,
            cfg.voltage_feedback_tag: 25.0,
            cfg.temp_tag: 50.0,
        }
        current_a, voltage_v, temp_c = svc._inputs_from_tags(tags)
        assert current_a == pytest.approx(100.0)
        assert voltage_v == pytest.approx(25.0)
        assert temp_c == pytest.approx(cfg.temp_full_scale_c / 2.0)


# --------------------------------------------------------------------------
# PID setpoint write path
# --------------------------------------------------------------------------


class TestPidSetpointWrite:
    """The service now delegates to the client's public write_pid_setpoint seam
    (the register-write details are tested against AsyncModbusManager in
    test_modbus_estop.py). These verify the delegation contract."""

    def test_delegates_to_client_seam(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            ok = await svc._write_pid_setpoint(2, 25.0)
            assert ok is True
            plc.write_pid_setpoint.assert_awaited_once_with(2, 25.0)

        asyncio.run(_go())

    def test_returns_client_result(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            plc.write_pid_setpoint = AsyncMock(return_value=False)
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            assert await svc._write_pid_setpoint(0, 50.0) is False

        asyncio.run(_go())


# --------------------------------------------------------------------------
# FastAPI router endpoints
# --------------------------------------------------------------------------


@pytest.fixture
def client_and_service(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, PowerSupplyService]:
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    plc = _FakePLC()
    svc = PowerSupplyService(plc, logging.getLogger("test"))
    app = FastAPI()
    app.include_router(create_power_supply_router(svc))
    return TestClient(app), svc


class TestRouterEndpoints:
    def test_get_config_returns_defaults_with_ramp_field(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, _ = client_and_service
        resp = client.get("/api/power_supply/config")
        assert resp.status_code == 200
        body = resp.json()
        assert body["current_full_scale_a"] == 200.0
        assert body["setpoint_ramp_rate_pct_per_s"] == 5.0

    def test_put_config_validates_and_replaces(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, svc = client_and_service
        new_cfg = svc.controller.config.model_dump()
        new_cfg["setpoint_ramp_rate_pct_per_s"] = 2.5
        new_cfg["current_setpoint_max_a"] = 30.0
        resp = client.put("/api/power_supply/config", json=new_cfg)
        assert resp.status_code == 200
        assert resp.json()["setpoint_ramp_rate_pct_per_s"] == 2.5
        assert svc.controller.config.current_setpoint_max_a == 30.0

    def test_put_config_invalid_returns_422(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, svc = client_and_service
        bad = svc.controller.config.model_dump()
        bad["setpoint_ramp_rate_pct_per_s"] = -1.0  # rejected by Pydantic
        resp = client.put("/api/power_supply/config", json=bad)
        assert resp.status_code == 422

    def test_get_status_returns_current_state(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, _ = client_and_service
        resp = client.get("/api/power_supply/status")
        assert resp.status_code == 200
        assert resp.json()["state"] == PowerSupplyState.IDLE

    def test_permissive_toggle(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, _ = client_and_service
        resp = client.post("/api/power_supply/permissive", json={"enabled": True})
        assert resp.status_code == 200
        assert resp.json()["state"] == PowerSupplyState.ARMED
        resp = client.post("/api/power_supply/permissive", json={"enabled": False})
        assert resp.json()["state"] == PowerSupplyState.IDLE

    def test_setpoint_current_mode_applies_and_clamps(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, _ = client_and_service
        client.post("/api/power_supply/permissive", json={"enabled": True})
        resp = client.post(
            "/api/power_supply/setpoint",
            json={"mode": "current", "value_a": 9999.0},
        )
        assert resp.status_code == 200
        # Clamped to the default current_setpoint_max_a (200 A).
        assert resp.json() == {"mode": "current", "applied_a": 200.0}

    def test_setpoint_current_mode_missing_value_returns_400(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, _ = client_and_service
        client.post("/api/power_supply/permissive", json={"enabled": True})
        resp = client.post(
            "/api/power_supply/setpoint",
            json={"mode": "current"},
        )
        assert resp.status_code == 400

    def test_setpoint_power_mode_missing_value_returns_400(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, _ = client_and_service
        resp = client.post(
            "/api/power_supply/setpoint",
            json={"mode": "power"},
        )
        assert resp.status_code == 400

    def test_setpoint_power_mode_negative_returns_400(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, _ = client_and_service
        client.post("/api/power_supply/permissive", json={"enabled": True})
        resp = client.post(
            "/api/power_supply/setpoint",
            json={"mode": "power", "value_w": -10.0},
        )
        assert resp.status_code == 400

    def test_acknowledge_trip_no_op_returns_status(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, _ = client_and_service
        resp = client.post("/api/power_supply/acknowledge_trip")
        assert resp.status_code == 200
        assert resp.json()["state"] == PowerSupplyState.IDLE


# --------------------------------------------------------------------------
# Setpoint request schema
# --------------------------------------------------------------------------


class TestSetpointRequestSchema:
    def test_invalid_mode_rejected_by_pydantic(
        self,
        client_and_service: tuple[TestClient, PowerSupplyService],
    ) -> None:
        client, _ = client_and_service
        resp = client.post(
            "/api/power_supply/setpoint",
            json={"mode": "bogus", "value_a": 5.0},
        )
        assert resp.status_code == 422
