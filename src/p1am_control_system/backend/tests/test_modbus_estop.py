"""Tests for AsyncModbusManager.trigger_estop best-effort kill semantics.

The kill must attempt EVERY zeroing write even when one fails (a partial kill is
unsafe), and report False so the caller retries.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from modbus_client import AsyncModbusManager  # noqa: E402


def _ok() -> MagicMock:
    return MagicMock(isError=lambda: False)


def _err() -> MagicMock:
    return MagicMock(isError=lambda: True)


def _manager_with(write_registers: AsyncMock) -> AsyncModbusManager:
    m = AsyncModbusManager(host="127.0.0.1")
    m._connected = True
    raw = MagicMock()
    raw.write_registers = write_registers
    m._get_client = lambda: raw  # type: ignore[method-assign]
    return m


# 4 PID setpoint writes + 1 tag-block write = 5 total.
_EXPECTED_WRITES = 5


class TestTriggerEstop:
    def test_all_writes_succeed_returns_true(self) -> None:
        async def go() -> None:
            wr = AsyncMock(return_value=_ok())
            m = _manager_with(wr)
            assert await m.trigger_estop() is True
            assert wr.await_count == _EXPECTED_WRITES

        asyncio.run(go())

    def test_one_failure_still_attempts_all_and_returns_false(self) -> None:
        async def go() -> None:
            # Fail the 2nd PID setpoint (address 212); everything else succeeds.
            def _side(*_: Any, address: int = 0, **__: Any) -> MagicMock:
                return _err() if address == 212 else _ok()

            wr = AsyncMock(side_effect=_side)
            m = _manager_with(wr)
            result = await m.trigger_estop()
            assert result is False  # reported as failed...
            assert wr.await_count == _EXPECTED_WRITES  # ...but ALL writes attempted

        asyncio.run(go())

    def test_disconnected_returns_false_without_writing(self) -> None:
        async def go() -> None:
            wr = AsyncMock(return_value=_ok())
            m = _manager_with(wr)
            m._connected = False
            assert await m.trigger_estop() is False
            assert wr.await_count == 0

        asyncio.run(go())


class TestWritePidSetpoint:
    """Register-write path for a PID setpoint (moved here from the PS service)."""

    def test_success_targets_correct_register(self) -> None:
        async def go() -> None:
            captured: dict[str, object] = {}

            def _side(*_: object, address: int = 0, values: object = None) -> object:
                captured["address"] = address
                captured["values"] = values
                return _ok()

            wr = AsyncMock(side_effect=_side)
            m = _manager_with(wr)
            assert await m.write_pid_setpoint(1, 25.0) is True
            assert captured["address"] == 212  # 200 + 1*10 + 2
            assert len(captured["values"]) == 2  # type: ignore[arg-type]

        asyncio.run(go())

    def test_disconnected_returns_false(self) -> None:
        async def go() -> None:
            wr = AsyncMock(return_value=_ok())
            m = _manager_with(wr)
            m._connected = False
            assert await m.write_pid_setpoint(0, 50.0) is False
            assert wr.await_count == 0

        asyncio.run(go())

    def test_invalid_index_returns_false_without_writing(self) -> None:
        async def go() -> None:
            wr = AsyncMock(return_value=_ok())
            m = _manager_with(wr)
            for bad in (-1, 4, 99):
                assert await m.write_pid_setpoint(bad, 50.0) is False
            assert wr.await_count == 0

        asyncio.run(go())

    def test_error_response_returns_false(self) -> None:
        async def go() -> None:
            m = _manager_with(AsyncMock(return_value=_err()))
            assert await m.write_pid_setpoint(0, 50.0) is False

        asyncio.run(go())

    def test_exception_returns_false(self) -> None:
        async def go() -> None:
            m = _manager_with(AsyncMock(side_effect=RuntimeError("dropped")))
            assert await m.write_pid_setpoint(0, 50.0) is False
            assert m._connected is False

        asyncio.run(go())
