"""F07 (#5528): the firmware first-out trip tag reaches the telemetry frame.

The firmware latches ``hardware.INTERLOCK_TRIPPED_REGISTER`` (561) and the
causing broker tag in ``hardware.INTERLOCK_TRIP_TAG_REGISTER`` (562). The host
reads both on the scan path and publishes ``first_out_tag_id``. Fail closed:
anything that is not a clean, latched, in-range tag is ``None`` (unknown) --
never a guessed value -- and a failed read must not disturb the scan.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pymodbus")
pytest.importorskip("sqlmodel")
pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).parent.parent))

import hardware  # noqa: E402
from main import poll_once  # noqa: E402
from modbus_client import AsyncModbusManager  # noqa: E402
from modbus_codec import decode_first_out_tag  # noqa: E402
from simulator_client import SimulatedPLCClient  # noqa: E402


class _Resp:
    def __init__(self, registers: list[int] | None = None, error: bool = False) -> None:
        self.registers = registers or []
        self._error = error

    def isError(self) -> bool:  # noqa: N802 - pymodbus API name
        return self._error


class _Transport:
    def __init__(self, registers: list[int] | None, *, raises: bool = False) -> None:
        self.registers = registers
        self.raises = raises
        self.reads: list[tuple[int, int]] = []

    async def read_holding_registers(self, address: int, count: int) -> _Resp:
        self.reads.append((address, count))
        if self.raises:
            raise OSError("link down")
        if self.registers is None:
            return _Resp(error=True)
        return _Resp(list(self.registers))


def _manager(transport: _Transport) -> AsyncModbusManager:
    m = AsyncModbusManager(host="127.0.0.1")
    m.client = transport  # type: ignore[assignment]
    m._connected = True
    return m


UNMAPPED = hardware.UNMAPPED_TAG_INDEX


@pytest.mark.parametrize(
    ("tripped", "tag", "expected"),
    [
        (1, 7, 7),
        (1, 0, 0),
        (1, hardware.TAG_COUNT - 1, hardware.TAG_COUNT - 1),
        (0, 7, None),  # not tripped: stale tag register is ignored
        (0, UNMAPPED, None),
        (1, UNMAPPED, None),  # tripped but cause unmapped -> unknown
        (1, hardware.TAG_COUNT, None),  # out of range -> unknown
        (2, 7, None),  # malformed latch flag -> unknown
    ],
)
def test_decode_first_out_tag_fails_closed(
    tripped: int, tag: int, expected: int | None
) -> None:
    assert decode_first_out_tag(tripped, tag) == expected


def test_read_first_out_tag_returns_tag_while_tripped() -> None:
    transport = _Transport([1, 5])
    m = _manager(transport)
    assert asyncio.run(m.read_first_out_tag()) == 5
    assert transport.reads == [(hardware.INTERLOCK_TRIPPED_REGISTER, 2)]


def test_read_first_out_tag_none_when_not_tripped() -> None:
    assert asyncio.run(_manager(_Transport([0, UNMAPPED])).read_first_out_tag()) is None


def test_read_first_out_tag_none_on_modbus_error_keeps_link() -> None:
    m = _manager(_Transport(None))
    assert asyncio.run(m.read_first_out_tag()) is None
    assert m.connected is True  # a status read must not drop the control link


def test_read_first_out_tag_none_on_exception_keeps_link() -> None:
    m = _manager(_Transport([1, 5], raises=True))
    assert asyncio.run(m.read_first_out_tag()) is None
    assert m.connected is True


def test_read_first_out_tag_none_when_disconnected() -> None:
    m = _manager(_Transport([1, 5]))
    m._connected = False
    assert asyncio.run(m.read_first_out_tag()) is None


def test_short_register_response_is_unknown() -> None:
    assert asyncio.run(_manager(_Transport([1])).read_first_out_tag()) is None


def test_simulator_never_fabricates_a_first_out_tag() -> None:
    sim = SimulatedPLCClient()
    assert asyncio.run(sim.read_first_out_tag()) is None


# ---- frame wiring ------------------------------------------------------


class _Status:
    def model_dump(self) -> dict[str, str]:
        return {"state": "ok"}


class _PLC:
    def __init__(self, first_out: Any = 4, *, raises: bool = False) -> None:
        self.connected = True
        self._first_out = first_out
        self._raises = raises
        self.estops = 0
        self.beats = 0

    async def read_tags(self) -> dict[str, float]:
        return {"TAG_0": 1.0}

    async def read_first_out_tag(self) -> int | None:
        if self._raises:
            raise RuntimeError("boom")
        return self._first_out

    async def trigger_estop(self) -> bool:
        self.estops += 1
        return True

    async def write_heartbeat(self) -> bool:
        self.beats += 1
        return True


class _Power:
    controller = type("C", (), {"config": type("Cfg", (), {"command_tag": "T"})()})()

    async def poll(self, tags: Any) -> _Status:
        return _Status()


class _Alicats:
    def get_devices_data(self) -> list[Any]:
        return []


class _Ws:
    async def broadcast(self, message: dict[str, Any]) -> None:
        return None


class _Alarms:
    def update_tag(self, name: str, value: float) -> list[Any]:
        return []


def _scan(plc: Any, *, estop: bool = False) -> dict[str, Any]:
    return asyncio.run(
        poll_once(
            plc=plc,
            latest_tag_values={},
            ws=_Ws(),
            alicats=_Alicats(),
            power_supply=_Power(),
            alarm_engine=_Alarms(),
            active_alarm_map={},
            estop_active=estop,
        )
    )


def test_frame_carries_first_out_tag_id() -> None:
    assert _scan(_PLC(first_out=4))["first_out_tag_id"] == 4


def test_frame_first_out_none_when_not_tripped() -> None:
    assert _scan(_PLC(first_out=None))["first_out_tag_id"] is None


def test_raising_client_does_not_break_scan_or_estop_path() -> None:
    plc = _PLC(raises=True)
    frame = _scan(plc, estop=True)
    assert frame["first_out_tag_id"] is None
    assert plc.estops == 1  # de-energize path ran regardless


def test_client_without_the_seam_yields_none() -> None:
    class _Bare(_PLC):
        read_first_out_tag = None  # type: ignore[assignment]

    assert _scan(_Bare())["first_out_tag_id"] is None


def test_non_int_first_out_from_client_is_rejected() -> None:
    assert _scan(_PLC(first_out="7"))["first_out_tag_id"] is None
