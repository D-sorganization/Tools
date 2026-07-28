"""Stop-path tests for the temperature service + router.

A Stop is the safety-critical direction: pressing it must de-energize the heater
relay immediately, not on the next scan. These cover both the service seam
(``set_permissive(False)`` writes the relay OFF now) and the FastAPI endpoint the
HMI actually calls. Kept in their own module so the main integration suite stays
under the repo's per-file size budget.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, cast
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
from temperature_models import TemperatureState  # noqa: E402


class _FakePLC:
    """Minimal PLC double — only the coil seam the service touches."""

    def __init__(self, *, connected: bool = True) -> None:
        self.connected = connected
        self.write_coil = AsyncMock(return_value=True)


def _service(plc: Any | None = None) -> TemperatureService:
    return TemperatureService(plc or _FakePLC(), logging.getLogger("test"))


@pytest.fixture
def client_and_service(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, TemperatureService]:
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    svc = _service()
    app = FastAPI()
    app.include_router(create_temperature_router(svc))
    return TestClient(app), svc


class TestImmediateStop:
    """service.set_permissive() de-energizes the heater the instant it's OFF."""

    def test_stop_writes_relay_off_now(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = _service(plc)
            svc.controller.set_permissive(True)
            svc.controller.set_setpoint_c(500.0)
            assert svc.controller.state == TemperatureState.RUNNING
            plc.write_coil.reset_mock()
            status = await svc.set_permissive(False)
            assert status.state == TemperatureState.IDLE
            # Relay commanded OFF immediately, without waiting for a poll.
            plc.write_coil.assert_any_await(hardware.HEATER_RELAY_COIL, False)

        asyncio.run(_go())

    def test_enable_arms_without_energizing(self) -> None:
        async def _go() -> None:
            plc = _FakePLC()
            svc = _service(plc)
            plc.write_coil.reset_mock()
            status = await svc.set_permissive(True)
            assert status.state == TemperatureState.ARMED
            # Enabling never commands the relay ON here (that stays with tick()).
            for call in plc.write_coil.await_args_list:
                if call.args[0] == hardware.HEATER_RELAY_COIL:
                    assert call.args[1] is False

        asyncio.run(_go())

    def test_rejects_non_bool(self) -> None:
        async def _go() -> None:
            svc = _service()
            with pytest.raises(TypeError):
                # cast so mypy accepts the arg under both full + skip modes; the
                # str still trips the runtime type guard we're asserting.
                await svc.set_permissive(cast(bool, "nope"))

        asyncio.run(_go())


class TestStopEndpoint:
    """POST /api/temperature/permissive {enabled:false} de-energizes now."""

    def test_stop_deenergizes_relay_immediately(
        self, client_and_service: tuple[TestClient, TemperatureService]
    ) -> None:
        client, svc = client_and_service
        client.post("/api/temperature/permissive", json={"enabled": True})
        client.post("/api/temperature/setpoint", json={"value_c": 500.0})
        assert svc.controller.state == TemperatureState.RUNNING
        svc._plc_client.write_coil.reset_mock()
        resp = client.post("/api/temperature/permissive", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["state"] == TemperatureState.IDLE
        svc._plc_client.write_coil.assert_any_await(hardware.HEATER_RELAY_COIL, False)
