"""Tests for AsyncModbusManager.trigger_estop best-effort kill semantics.

The kill must attempt EVERY de-energizing write even when one fails (a partial
kill is unsafe), and report False so the caller retries.

These assert on the ADDRESSES AND VALUES actually put on the wire, not on a
write count (issue #4033). A count assertion cannot distinguish "de-energized
the heater" from "wrote five arbitrary registers", and the old fake did not
even wire ``write_coil`` — so the heater relay, the ONE thing that commands the
110 V element, was literally unobservable.
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

import hardware  # noqa: E402
from modbus_client import AsyncModbusManager  # noqa: E402
from modbus_codec import float_to_registers  # noqa: E402


def _ok() -> MagicMock:
    return MagicMock(isError=lambda: False)


def _err() -> MagicMock:
    return MagicMock(isError=lambda: True)


class _EstopRecorder:
    """Fake pymodbus client recording register AND coil writes, in order.

    Both seams are wired so a missing heater-relay de-energize is observable.
    """

    def __init__(
        self,
        *,
        failing_register: int | None = None,
        failing_coil: int | None = None,
    ) -> None:
        self.register_writes: list[tuple[int, list[int]]] = []
        self.coil_writes: list[tuple[int, bool]] = []
        self.order: list[str] = []
        self._failing_register = failing_register
        self._failing_coil = failing_coil

    async def write_registers(self, address: int, values: list[int]) -> MagicMock:
        self.register_writes.append((address, list(values)))
        self.order.append(f"registers@{address}")
        return _err() if address == self._failing_register else _ok()

    async def write_coil(self, address: int, value: bool) -> MagicMock:
        self.coil_writes.append((address, value))
        self.order.append(f"coil@{address}")
        return _err() if address == self._failing_coil else _ok()


def _manager_with_recorder(rec: _EstopRecorder) -> AsyncModbusManager:
    m = AsyncModbusManager(host="127.0.0.1")
    m._connected = True
    m._get_client = lambda: rec
    return m


def _manager_with(write_registers: AsyncMock) -> AsyncModbusManager:
    m = AsyncModbusManager(host="127.0.0.1")
    m._connected = True
    raw = MagicMock()
    raw.write_registers = write_registers
    raw.write_coil = AsyncMock(return_value=_ok())
    m._get_client = lambda: raw
    return m


_ZERO = float_to_registers(0.0)
_PID_SETPOINT_ADDRESSES = [hardware.pid_setpoint_address(i) for i in range(4)]


class TestTriggerEstop:
    def test_opens_heater_relay_first(self) -> None:
        """The heater relay is the ONLY thing commanding the 110 V element.

        It must be de-energized as the FIRST wire action of an E-stop, not left
        to the next TemperatureService.poll() — which is seconds to tens of
        seconds away, or never if the poll loop is wedged (issue #4000).
        """

        async def go() -> None:
            rec = _EstopRecorder()
            m = _manager_with_recorder(rec)
            assert await m.trigger_estop() is True
            assert rec.coil_writes == [(hardware.HEATER_RELAY_COIL, False)]
            assert rec.order[0] == f"coil@{hardware.HEATER_RELAY_COIL}"

        asyncio.run(go())

    def test_zeroes_every_pid_setpoint_register(self) -> None:
        async def go() -> None:
            rec = _EstopRecorder()
            m = _manager_with_recorder(rec)
            assert await m.trigger_estop() is True
            assert rec.register_writes == [
                (address, _ZERO) for address in _PID_SETPOINT_ADDRESSES
            ]

        asyncio.run(go())

    def test_does_not_write_the_republished_tag_block(self) -> None:
        """The 64-register tag-block write was a provable no-op (issue #4000).

        The firmware rewrites registers 0..63 from its broker at the end of
        every scan and never reads them back, so the host's zeros were
        overwritten within one scan and never observed. Writing them wasted a
        Modbus transaction on the E-stop path and made the write count look
        reassuring.
        """

        async def go() -> None:
            rec = _EstopRecorder()
            m = _manager_with_recorder(rec)
            await m.trigger_estop()
            tag_block_end = hardware.TAG_VALUE_BASE + hardware.TAG_COUNT * 2
            for address, _values in rec.register_writes:
                assert not (hardware.TAG_VALUE_BASE <= address < tag_block_end), (
                    f"E-stop wrote {address}, inside the firmware-republished "
                    "tag block — a no-op the firmware overwrites every scan"
                )

        asyncio.run(go())

    def test_failed_relay_write_is_a_hard_error(self) -> None:
        async def go() -> None:
            rec = _EstopRecorder(failing_coil=hardware.HEATER_RELAY_COIL)
            m = _manager_with_recorder(rec)
            assert await m.trigger_estop() is False
            # ...and the setpoints were still zeroed (partial kill > no kill).
            assert len(rec.register_writes) == len(_PID_SETPOINT_ADDRESSES)

        asyncio.run(go())

    def test_one_failure_still_attempts_all_and_returns_false(self) -> None:
        async def go() -> None:
            # Fail the 2nd PID setpoint (address 212); everything else succeeds.
            rec = _EstopRecorder(failing_register=212)
            m = _manager_with_recorder(rec)
            assert await m.trigger_estop() is False  # reported as failed...
            # ...but ALL de-energizing writes were still attempted.
            assert rec.coil_writes == [(hardware.HEATER_RELAY_COIL, False)]
            assert [a for a, _ in rec.register_writes] == _PID_SETPOINT_ADDRESSES

        asyncio.run(go())

    def test_disconnected_returns_false_without_writing(self) -> None:
        async def go() -> None:
            rec = _EstopRecorder()
            m = _manager_with_recorder(rec)
            m._connected = False
            assert await m.trigger_estop() is False
            assert rec.order == []

        asyncio.run(go())

    def test_exception_returns_false_and_drops_connection(self) -> None:
        async def go() -> None:
            m = AsyncModbusManager(host="127.0.0.1")
            m._connected = True
            raw = MagicMock()
            raw.write_coil = AsyncMock(side_effect=RuntimeError("socket gone"))
            raw.write_registers = AsyncMock(side_effect=RuntimeError("socket gone"))
            m._get_client = lambda: raw
            assert await m.trigger_estop() is False
            assert m._connected is False

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


def _manager_with_coil(write_coil: AsyncMock) -> AsyncModbusManager:
    m = AsyncModbusManager(host="127.0.0.1")
    m._connected = True
    raw = MagicMock()
    raw.write_coil = write_coil
    m._get_client = lambda: raw
    return m


class TestWriteCoil:
    """Discrete-coil write path (e.g. the heater relay)."""

    def test_success_passes_address_and_value(self) -> None:
        async def go() -> None:
            captured: dict[str, object] = {}

            def _side(*_: Any, address: int = 0, value: bool = False) -> MagicMock:
                captured["address"] = address
                captured["value"] = value
                return _ok()

            wc = AsyncMock(side_effect=_side)
            m = _manager_with_coil(wc)
            assert await m.write_coil(2, True) is True
            assert captured == {"address": 2, "value": True}

        asyncio.run(go())

    def test_disconnected_returns_false_without_writing(self) -> None:
        async def go() -> None:
            wc = AsyncMock(return_value=_ok())
            m = _manager_with_coil(wc)
            m._connected = False
            assert await m.write_coil(2, True) is False
            assert wc.await_count == 0

        asyncio.run(go())

    def test_non_bool_value_raises(self) -> None:
        async def go() -> None:
            m = _manager_with_coil(AsyncMock(return_value=_ok()))
            with pytest.raises(TypeError):
                await m.write_coil(2, 1)

        asyncio.run(go())

    def test_error_response_returns_false(self) -> None:
        async def go() -> None:
            m = _manager_with_coil(AsyncMock(return_value=_err()))
            assert await m.write_coil(2, True) is False

        asyncio.run(go())

    def test_exception_marks_disconnected(self) -> None:
        async def go() -> None:
            m = _manager_with_coil(AsyncMock(side_effect=RuntimeError("dropped")))
            assert await m.write_coil(2, False) is False
            assert m._connected is False

        asyncio.run(go())
