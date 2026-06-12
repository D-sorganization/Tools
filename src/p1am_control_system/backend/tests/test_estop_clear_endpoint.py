"""Endpoint-level coverage for clearing a latched E-stop."""

from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest

os.environ["PLC_DRIVER"] = "modbus"
os.environ["P1AM_DEV_NO_AUTH"] = "1"

pytest.importorskip("sqlmodel")
pytest.importorskip("tools_core")
pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient
from main import app, modbus_manager

client = TestClient(app)


@pytest.fixture(autouse=True)
def restore_estop_latch() -> Generator[None, None, None]:
    """Keep the process-global E-stop latch isolated between tests."""
    import main as backend_main

    backend_main.e_stop_active = False
    try:
        yield
    finally:
        backend_main.e_stop_active = False


@pytest.mark.asyncio
async def test_estop_clear_commands_plc_when_connected() -> None:
    """POST /api/estop/clear MUST command the connected PLC to reset (#3314)."""
    import main as backend_main

    mock_clear = AsyncMock(return_value=True)
    mock_backup_clear = AsyncMock(return_value=True)

    with (
        patch.object(modbus_manager, "_connected", True),
        patch.object(modbus_manager, "clear_estop", mock_clear),
        patch.object(backend_main.backup_simulator, "clear_estop", mock_backup_clear),
    ):
        backend_main.e_stop_active = True
        response = client.post("/api/estop/clear")

        assert response.status_code == 200
        assert "cleared" in response.json()["message"].lower()
        # The controller reset MUST have been commanded, not just a local flag.
        mock_clear.assert_called_once()
        assert backend_main.e_stop_active is False


@pytest.mark.asyncio
async def test_estop_clear_keeps_latch_when_plc_rejects() -> None:
    """If the PLC does not acknowledge the reset, the HMI must stay tripped."""
    import main as backend_main

    mock_clear = AsyncMock(return_value=False)

    with (
        patch.object(modbus_manager, "_connected", True),
        patch.object(modbus_manager, "clear_estop", mock_clear),
    ):
        backend_main.e_stop_active = True
        response = client.post("/api/estop/clear")

        assert response.status_code == 502
        mock_clear.assert_called_once()
        # Latch preserved: HMI keeps showing the tripped state.
        assert backend_main.e_stop_active is True


@pytest.mark.asyncio
async def test_estop_clear_uses_backup_when_plc_offline() -> None:
    """With the PLC offline the backup simulator handles and confirms the reset."""
    import main as backend_main

    mock_backup_clear = AsyncMock(return_value=True)

    with (
        patch.object(modbus_manager, "_connected", False),
        patch.object(backend_main.backup_simulator, "clear_estop", mock_backup_clear),
    ):
        backend_main.e_stop_active = True
        response = client.post("/api/estop/clear")

        assert response.status_code == 200
        assert "Simulated" in response.json()["message"]
        mock_backup_clear.assert_called_once()
        assert backend_main.e_stop_active is False
