"""Client-level coverage for the E-stop *reset* path (issue #3314).

The header showed a green "E-STOP CLEAR" while the plant stayed tripped because
clearing only flipped a server-side flag and never commanded the controller.
These tests pin the contract that each PLC client must issue (and confirm) a real
reset command before reporting success.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("pymodbus")

from modbus_client import AsyncModbusManager  # noqa: E402
from simulator_client import SimulatedPLCClient  # noqa: E402


class _NullAsyncLock:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: Any) -> None:
        return None


def _make_manager(*, connected: bool, is_error: bool = False) -> AsyncModbusManager:
    mgr = AsyncModbusManager(host="127.0.0.1")
    mgr._connected = connected
    mgr.lock = _NullAsyncLock()  # type: ignore[assignment]
    raw = MagicMock()
    raw.write_coil = AsyncMock(return_value=MagicMock(isError=lambda: is_error))
    mgr.client = raw
    return mgr


def test_modbus_clear_estop_pulses_reset_coil() -> None:
    """A connected PLC must receive a reset-coil write and report success."""

    async def _go() -> None:
        mgr = _make_manager(connected=True)
        result = await mgr.clear_estop()
        assert result is True
        mgr.client.write_coil.assert_awaited_once()
        kwargs = mgr.client.write_coil.await_args.kwargs
        assert kwargs["address"] == mgr.estop_reset_coil_address
        assert kwargs["value"] is True

    asyncio.run(_go())


def test_modbus_clear_estop_false_when_disconnected() -> None:
    """A disconnected PLC cannot confirm a reset, so clear must fail."""

    async def _go() -> None:
        mgr = _make_manager(connected=False)
        result = await mgr.clear_estop()
        assert result is False
        mgr.client.write_coil.assert_not_awaited()

    asyncio.run(_go())


def test_modbus_clear_estop_false_on_error_response() -> None:
    """An error response from the controller must surface as a failed clear."""

    async def _go() -> None:
        mgr = _make_manager(connected=True, is_error=True)
        result = await mgr.clear_estop()
        assert result is False
        # The reset was attempted but the controller rejected it.
        mgr.client.write_coil.assert_awaited_once()

    asyncio.run(_go())


def test_modbus_clear_estop_false_on_exception() -> None:
    """A transport exception must fail the clear and drop the connection."""

    async def _go() -> None:
        mgr = _make_manager(connected=True)
        mgr.client.write_coil = AsyncMock(side_effect=RuntimeError("link down"))
        result = await mgr.clear_estop()
        assert result is False
        assert mgr.connected is False

    asyncio.run(_go())


def test_simulator_clear_estop_resets_latch() -> None:
    """The simulator must clear its own latched E-stop state on reset."""

    async def _go() -> None:
        sim = SimulatedPLCClient()
        await sim.trigger_estop()
        assert sim.e_stop_active is True
        result = await sim.clear_estop()
        assert result is True
        assert sim.e_stop_active is False

    asyncio.run(_go())


def test_simulator_write_coil_records_state() -> None:
    """The simulator must record discrete-coil writes (e.g. the heater relay)."""

    async def _go() -> None:
        sim = SimulatedPLCClient()
        assert await sim.write_coil(2, True) is True
        assert sim.coils[2] is True
        assert await sim.write_coil(2, False) is True
        assert sim.coils[2] is False

    asyncio.run(_go())
