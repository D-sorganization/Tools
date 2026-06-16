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
    def test_poll_with_no_tags_uses_zero_feedback(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            status = await svc.poll(None)
            assert status.measured_current_a == 0.0
            assert status.measured_voltage_v == 0.0
            assert status.measured_temp_c == 0.0
            assert status.state == PowerSupplyState.IDLE

        asyncio.run(_go())

    def test_poll_writes_pid_setpoint_when_running(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            svc.controller.set_permissive(True)
            svc.controller.set_current_setpoint(40.0)
            await svc.poll({})
            await svc.poll({})
            call_args_list = plc._get_client().write_registers.await_args_list
            assert len(call_args_list) >= 1
            last_call = call_args_list[-1]
            assert last_call.kwargs["address"] == 202
            assert len(last_call.kwargs["values"]) == 2

        asyncio.run(_go())

    def test_inputs_from_tags_scales_percent_to_engineering(self) -> None:
        plc = _FakePLC()
        svc = PowerSupplyService(plc, logging.getLogger("test"))
        cfg = PowerSupplyConfig(current_full_scale_a=200.0, voltage_full_scale_v=100.0)
        svc.controller.update_config(cfg)
        # 50 % on current feedback tag → 100 A
        # 25 % on voltage feedback tag → 25 V
        # Temp passes through unchanged
        tags = {
            cfg.current_feedback_tag: 50.0,
            cfg.voltage_feedback_tag: 25.0,
            cfg.temp_tag: 800.0,
        }
        current_a, voltage_v, temp_c = svc._inputs_from_tags(tags)
        assert current_a == pytest.approx(100.0)
        assert voltage_v == pytest.approx(25.0)
        assert temp_c == pytest.approx(800.0)


# --------------------------------------------------------------------------
# PID setpoint write path
# --------------------------------------------------------------------------


class TestPidSetpointWrite:
    def test_write_when_disconnected_returns_false(self) -> None:
        async def _go() -> None:
            plc = _FakePLC(connected=False)
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            ok = await svc._write_pid_setpoint(0, 50.0)
            assert ok is False

        asyncio.run(_go())

    def test_write_with_invalid_pid_index_returns_false(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            for bad in (-1, 4, 5, 99):
                ok = await svc._write_pid_setpoint(bad, 50.0)
                assert ok is False

        asyncio.run(_go())

    def test_write_with_error_response_returns_false(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            plc._get_client().write_registers = AsyncMock(
                return_value=MagicMock(isError=lambda: True)
            )
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            ok = await svc._write_pid_setpoint(0, 50.0)
            assert ok is False

        asyncio.run(_go())

    def test_write_catches_underlying_exception(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            plc._get_client().write_registers = AsyncMock(
                side_effect=RuntimeError("modbus dropped")
            )
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            ok = await svc._write_pid_setpoint(0, 50.0)
            assert ok is False

        asyncio.run(_go())

    def test_write_success_returns_true_and_targets_pid_register(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            # PID 1 setpoint sits at register 200 + 1*10 + 2 = 212
            ok = await svc._write_pid_setpoint(1, 25.0)
            assert ok is True
            plc._get_client().write_registers.assert_awaited()
            kwargs = plc._get_client().write_registers.await_args.kwargs
            assert kwargs["address"] == 212

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
        assert body["current_full_scale_a"] == 100.0
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
        # Clamped to default max (50)
        assert resp.json() == {"mode": "current", "applied_a": 50.0}

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
