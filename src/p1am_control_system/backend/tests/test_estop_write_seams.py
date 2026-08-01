"""Contract tests for the AsyncModbusManager write seams.

Covers three defects that all share a root cause — a write seam that reports
success without reaching the plant:

* #4015 ``write_tag`` resolves ``TAG_n`` to holding register ``n*2``, inside the
  block the firmware republishes from its broker every scan and never reads
  back. The write can never influence the plant, yet it returned ``True``.
* #4038 ``write_tag`` did not consult the defense-in-depth ``_estop_active``
  latch that ``write_coil`` and ``write_pid_setpoint`` honour.
* #3999 the firmware watchdog (holding register 560) needs a host heartbeat
  seam; without a bump every scan the firmware drives outputs safe after 2 s.

The latch coverage is written as a registry so a NEW public write seam cannot
be added without either honouring the latch or being explicitly exempted here.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

import hardware  # noqa: E402
from modbus_client import AsyncModbusManager  # noqa: E402
from modbus_codec import float_to_registers, registers_to_float  # noqa: E402
from models import InterlockConfig, PIDConfig, RoutingConfig  # noqa: E402


def _ok() -> MagicMock:
    return MagicMock(isError=lambda: False)


class _Recorder:
    """Fake pymodbus client recording every register and coil write."""

    def __init__(self) -> None:
        self.register_writes: list[tuple[int, list[int]]] = []
        self.coil_writes: list[tuple[int, bool]] = []

    async def write_registers(self, address: int, values: list[int]) -> MagicMock:
        self.register_writes.append((address, list(values)))
        return _ok()

    async def write_coil(self, address: int, value: bool) -> MagicMock:
        self.coil_writes.append((address, value))
        return _ok()


def _manager(rec: _Recorder | None = None) -> AsyncModbusManager:
    m = AsyncModbusManager(host="127.0.0.1")
    m._connected = True
    m._get_client = lambda: rec or _Recorder()
    return m


def _v_register_tag_map(register_num: int = 1000) -> dict[str, Any]:
    """A dynamic project tag backed by a real, host-writable V register."""
    return {
        "PUMP_CMD": SimpleNamespace(register_type="V", register_num=register_num),
    }


# ---------------------------------------------------------------------------
# #4015 — write_tag must not claim a write it cannot perform
# ---------------------------------------------------------------------------


class TestWriteTagFailsLoudly:
    def test_tag_block_write_raises_instead_of_reporting_success(self) -> None:
        """``TAG_n`` lands in the firmware-republished block: refuse it.

        Returning ``True`` here made the API answer 200 for a command the plant
        never saw, and let the PID auto-tuner fit gains to a step that never
        happened.
        """

        async def go() -> None:
            rec = _Recorder()
            m = _manager(rec)
            with pytest.raises(NotImplementedError) as excinfo:
                await m.write_tag("TAG_5", 42.5)
            assert "TAG_5" in str(excinfo.value)
            assert rec.register_writes == []

        asyncio.run(go())

    def test_every_broker_tag_is_refused(self) -> None:
        async def go() -> None:
            m = _manager()
            for index in range(hardware.TAG_COUNT):
                with pytest.raises(NotImplementedError):
                    await m.write_tag(hardware.tag_name(index), 1.0)

        asyncio.run(go())

    def test_mapped_v_register_tag_still_writes(self) -> None:
        """A dynamic V-register tag is a real host-writable address; keep it."""

        async def go() -> None:
            rec = _Recorder()
            m = _manager(rec)
            m.tag_map = _v_register_tag_map()
            assert await m.write_tag("PUMP_CMD", 12.5) is True
            assert rec.register_writes == [(1000, float_to_registers(12.5))]

        asyncio.run(go())

    def test_unresolvable_tag_returns_false_without_writing(self) -> None:
        async def go() -> None:
            rec = _Recorder()
            m = _manager(rec)
            assert await m.write_tag("NOT_A_TAG", 1.0) is False
            assert rec.register_writes == []

        asyncio.run(go())


# ---------------------------------------------------------------------------
# #3999 — firmware watchdog heartbeat
# ---------------------------------------------------------------------------


class TestWriteHeartbeat:
    def test_targets_the_firmware_heartbeat_register(self) -> None:
        async def go() -> None:
            rec = _Recorder()
            m = _manager(rec)
            assert await m.write_heartbeat() is True
            assert len(rec.register_writes) == 1
            address, values = rec.register_writes[0]
            assert address == hardware.HOST_HEARTBEAT_REGISTER
            assert len(values) == 1

        asyncio.run(go())

    def test_value_changes_every_call(self) -> None:
        """The firmware proves liveness from a CHANGE, not from a value."""

        async def go() -> None:
            rec = _Recorder()
            m = _manager(rec)
            for _ in range(5):
                assert await m.write_heartbeat() is True
            written = [values[0] for _address, values in rec.register_writes]
            assert len(set(written)) == len(written)

        asyncio.run(go())

    def test_counter_wraps_inside_16_bits(self) -> None:
        async def go() -> None:
            rec = _Recorder()
            m = _manager(rec)
            m._heartbeat_counter = 0xFFFF
            assert await m.write_heartbeat() is True
            assert await m.write_heartbeat() is True
            for _address, values in rec.register_writes:
                assert 0 <= values[0] <= 0xFFFF

        asyncio.run(go())

    def test_disconnected_returns_false_without_writing(self) -> None:
        async def go() -> None:
            rec = _Recorder()
            m = _manager(rec)
            m._connected = False
            assert await m.write_heartbeat() is False
            assert rec.register_writes == []

        asyncio.run(go())

    def test_exception_drops_the_connection(self) -> None:
        async def go() -> None:
            m = AsyncModbusManager(host="127.0.0.1")
            m._connected = True
            raw = MagicMock()
            raw.write_registers = AsyncMock(side_effect=RuntimeError("socket gone"))
            m._get_client = lambda: raw
            assert await m.write_heartbeat() is False
            assert m._connected is False

        asyncio.run(go())

    def test_exposed_on_the_base_client_interface(self) -> None:
        """Callers (the poll loop) address the seam through BasePLCClient."""
        from plc_interface import BasePLCClient

        assert hasattr(BasePLCClient, "write_heartbeat")


# ---------------------------------------------------------------------------
# #4038 — EVERY public write seam honours the E-stop latch
# ---------------------------------------------------------------------------


def _routing_with_hot_setpoints() -> RoutingConfig:
    return RoutingConfig(
        input_routing=[f"TAG_{i}" for i in range(6)],
        output_routing=["TAG_10", "TAG_11"],
        pids=[
            PIDConfig(
                pv_tag="TAG_1", cv_tag="TAG_2", setpoint=80.0, kp=1.0, ki=0.5, kd=0.1
            )
            for _ in range(4)
        ],
        interlocks={
            f"TAG_{i}": InterlockConfig(
                lolo_limit=0.0, low_limit=5.0, high_limit=95.0, hihi_limit=100.0
            )
            for i in range(32)
        },
    )


async def _check_write_coil(m: AsyncModbusManager, rec: _Recorder) -> None:
    await m.write_coil(hardware.HEATER_RELAY_COIL, True)
    assert rec.coil_writes == [(hardware.HEATER_RELAY_COIL, False)]


async def _check_write_pid_setpoint(m: AsyncModbusManager, rec: _Recorder) -> None:
    await m.write_pid_setpoint(0, 75.0)
    assert rec.register_writes == [(hardware.pid_setpoint_address(0), [0, 0])]


async def _check_write_tag(m: AsyncModbusManager, rec: _Recorder) -> None:
    m.tag_map = _v_register_tag_map()
    await m.write_tag("PUMP_CMD", 75.0)
    assert [address for address, _ in rec.register_writes] == [1000]
    low, high = rec.register_writes[0][1]
    assert registers_to_float(low, high) == 0.0


async def _check_write_routing(m: AsyncModbusManager, rec: _Recorder) -> None:
    """A routing deploy carries PID setpoints — it must not re-energize."""
    assert await m.write_routing(_routing_with_hot_setpoints()) is False
    assert rec.register_writes == []


# name -> behavioural assertion that the latch forced the safe direction.
_LATCH_CHECKS = {
    "write_coil": _check_write_coil,
    "write_pid_setpoint": _check_write_pid_setpoint,
    "write_tag": _check_write_tag,
    "write_routing": _check_write_routing,
}

# Seams deliberately NOT gated by the latch, each with a safety rationale.
_LATCH_EXEMPT = {
    # A liveness counter, not a plant output. Suppressing it while the E-stop
    # is latched would make the firmware's 2 s watchdog trip on a host that is
    # in fact alive and deliberately holding the plant down — and would mask a
    # genuine host failure behind an operator action.
    "write_heartbeat",
}


@pytest.mark.parametrize("seam_name", sorted(_LATCH_CHECKS))
def test_public_write_seam_honours_estop_latch(seam_name: str) -> None:
    async def go() -> None:
        rec = _Recorder()
        m = _manager(rec)
        m.set_estop_active(True)
        await _LATCH_CHECKS[seam_name](m, rec)

    asyncio.run(go())


def test_no_public_write_seam_escapes_the_latch_registry() -> None:
    """Guards the seam surface itself, not just today's four methods.

    A new ``write_*`` method added to the manager fails here until it is either
    covered by a latch assertion above or explicitly exempted with a reason.
    """
    seams = {
        name
        for name, member in inspect.getmembers(
            AsyncModbusManager, inspect.iscoroutinefunction
        )
        if name.startswith("write_")
    }
    uncovered = seams - set(_LATCH_CHECKS) - _LATCH_EXEMPT
    assert not uncovered, (
        f"public write seam(s) {sorted(uncovered)} are not covered by the "
        "E-stop latch registry — add a latch assertion or an explicit exemption"
    )
