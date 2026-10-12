"""F07 (#5528): the firmware trip latch and first-out tag reach the telemetry frame.

The firmware latches ``hardware.INTERLOCK_TRIPPED_REGISTER`` (561) and the
causing broker tag in ``hardware.INTERLOCK_TRIP_TAG_REGISTER`` (562). The host
reads both in ONE register read on the scan path and publishes an explicit
tri-state ``plc_trip_latch`` ("tripped" / "clear" / "unknown") plus
``first_out_tag_id``. Fail closed: a failed or malformed read is "unknown",
never "clear", and a tag is only ever an int while the latch is tripped.
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
from hardware import FIRST_OUT_UNKNOWN, FirstOutStatus  # noqa: E402
from main import poll_once  # noqa: E402
from modbus_client import AsyncModbusManager  # noqa: E402
from modbus_codec import decode_first_out_status  # noqa: E402
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
    ("tripped", "tag", "latch", "tag_id"),
    [
        (1, 7, "tripped", 7),
        (1, 0, "tripped", 0),
        (1, hardware.TAG_COUNT - 1, "tripped", hardware.TAG_COUNT - 1),
        (0, 7, "clear", None),  # not tripped: stale tag register is ignored
        (0, UNMAPPED, "clear", None),
        (1, UNMAPPED, "tripped", None),  # latched, cause unknown
        (1, hardware.TAG_COUNT, "tripped", None),  # out-of-range tag
        (2, 7, "unknown", None),  # malformed latch flag
        (-1, 7, "unknown", None),
    ],
)
def test_decode_first_out_status_fails_closed(
    tripped: int, tag: int, latch: str, tag_id: int | None
) -> None:
    assert decode_first_out_status(tripped, tag) == FirstOutStatus(latch, tag_id)


def test_unknown_constant_is_unknown_with_no_tag() -> None:
    assert FIRST_OUT_UNKNOWN == FirstOutStatus("unknown", None)


def test_read_status_tripped_with_tag_is_a_single_read() -> None:
    transport = _Transport([1, 5])
    m = _manager(transport)
    assert asyncio.run(m.read_first_out_status()) == FirstOutStatus("tripped", 5)
    assert transport.reads == [(hardware.INTERLOCK_TRIPPED_REGISTER, 2)]


def test_read_status_clear() -> None:
    m = _manager(_Transport([0, UNMAPPED]))
    assert asyncio.run(m.read_first_out_status()) == FirstOutStatus("clear", None)


def test_read_status_tripped_unmapped_tag() -> None:
    m = _manager(_Transport([1, UNMAPPED]))
    assert asyncio.run(m.read_first_out_status()) == FirstOutStatus("tripped", None)


def test_read_status_unknown_on_modbus_error_keeps_link() -> None:
    m = _manager(_Transport(None))
    assert asyncio.run(m.read_first_out_status()) == FIRST_OUT_UNKNOWN
    assert m.connected is True  # a status read must not drop the control link


def test_read_status_unknown_on_exception_keeps_link() -> None:
    m = _manager(_Transport([1, 5], raises=True))
    assert asyncio.run(m.read_first_out_status()) == FIRST_OUT_UNKNOWN
    assert m.connected is True


def test_read_status_unknown_when_disconnected() -> None:
    m = _manager(_Transport([1, 5]))
    m._connected = False
    assert asyncio.run(m.read_first_out_status()) == FIRST_OUT_UNKNOWN


def test_short_register_response_is_unknown() -> None:
    m = _manager(_Transport([1]))
    assert asyncio.run(m.read_first_out_status()) == FIRST_OUT_UNKNOWN


def test_simulator_reports_unknown_never_fabricated() -> None:
    sim = SimulatedPLCClient()
    assert asyncio.run(sim.read_first_out_status()) == FIRST_OUT_UNKNOWN


# ---- frame wiring ------------------------------------------------------


class _Status:
    def model_dump(self) -> dict[str, str]:
        return {"state": "ok"}


_DEFAULT_STATUS = FirstOutStatus("tripped", 4)


class _PLC:
    def __init__(
        self,
        first_out: Any = _DEFAULT_STATUS,
        *,
        raises: bool = False,
    ) -> None:
        self.connected = True
        self._first_out = first_out
        self._raises = raises
        self.estops = 0

    async def read_tags(self) -> dict[str, float]:
        return {"TAG_0": 1.0}

    async def read_first_out_status(self) -> Any:
        if self._raises:
            raise RuntimeError("boom")
        return self._first_out

    async def trigger_estop(self) -> bool:
        self.estops += 1
        return True

    async def write_heartbeat(self) -> bool:
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


def test_frame_carries_latch_and_first_out_tag_id() -> None:
    frame = _scan(_PLC(first_out=FirstOutStatus("tripped", 4)))
    assert frame["first_out_tag_id"] == 4
    assert frame["plc_trip_latch"] == "tripped"


def test_frame_clear_has_no_tag() -> None:
    frame = _scan(_PLC(first_out=FirstOutStatus("clear", None)))
    assert frame["first_out_tag_id"] is None
    assert frame["plc_trip_latch"] == "clear"


def test_frame_tripped_with_unknown_tag() -> None:
    frame = _scan(_PLC(first_out=FirstOutStatus("tripped", None)))
    assert frame["plc_trip_latch"] == "tripped"
    assert frame["first_out_tag_id"] is None


def test_raising_client_is_unknown_and_estop_path_still_runs() -> None:
    plc = _PLC(raises=True)
    frame = _scan(plc, estop=True)
    assert frame["plc_trip_latch"] == "unknown"
    assert frame["first_out_tag_id"] is None
    assert plc.estops == 1  # de-energize path ran regardless


def test_client_without_the_seam_is_unknown() -> None:
    class _Bare(_PLC):
        read_first_out_status = None  # type: ignore[assignment]

    assert _scan(_Bare())["plc_trip_latch"] == "unknown"


@pytest.mark.parametrize(
    "bad", [7, "7", None, FirstOutStatus("bogus", 3), FirstOutStatus("clear", 3)]
)
def test_malformed_client_result_is_unknown(bad: Any) -> None:
    frame = _scan(_PLC(first_out=bad))
    assert frame["plc_trip_latch"] == "unknown"
    assert frame["first_out_tag_id"] is None
