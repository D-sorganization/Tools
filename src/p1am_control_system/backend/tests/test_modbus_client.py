"""Unit tests for ``AsyncModbusManager`` read/write/routing orchestration.

These exercise the highest-risk transport-layer logic — the 32-tag float
decode, the chunked routing read/reassembly, the routing write (including the
chunked interlock block), direct tag-address resolution, and the public
``write_pid_setpoint`` seam — against a fake pymodbus client. No PLC, no
network, and no ``tools_core``/``fastapi`` import is required (issue #3535).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pymodbus")
pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from modbus_client import AsyncModbusManager  # noqa: E402
from modbus_codec import float_to_registers  # noqa: E402
from models import (  # noqa: E402
    InterlockConfig,
    PIDConfig,
    RoutingConfig,
)


class _FakeResponse:
    """Minimal stand-in for a pymodbus response object."""

    def __init__(self, registers: list[int] | None = None, error: bool = False) -> None:
        self.registers = registers or []
        self._error = error

    def isError(self) -> bool:  # noqa: N802 - pymodbus API name
        return self._error


class _FakeModbusClient:
    """Records writes and serves scripted register reads for a fake PLC."""

    def __init__(self) -> None:
        self.read_responses: list[_FakeResponse] = []
        self.write_calls: list[dict[str, Any]] = []
        self.coil_calls: list[dict[str, Any]] = []
        self.force_read_error = False
        self.force_write_error = False

    async def read_holding_registers(self, address: int, count: int) -> _FakeResponse:
        if self.force_read_error:
            return _FakeResponse(error=True)
        if self.read_responses:
            return self.read_responses.pop(0)
        # Default: zeros of requested width.
        return _FakeResponse([0] * count)

    async def write_registers(self, address: int, values: list[int]) -> _FakeResponse:
        self.write_calls.append({"address": address, "values": list(values)})
        return _FakeResponse(error=self.force_write_error)

    async def write_coil(self, address: int, value: bool) -> _FakeResponse:
        self.coil_calls.append({"address": address, "value": value})
        return _FakeResponse(error=self.force_write_error)


def _make_manager(fake: _FakeModbusClient) -> AsyncModbusManager:
    manager = AsyncModbusManager(host="127.0.0.1")
    manager.client = fake  # inject fake transport
    manager._connected = True
    return manager


def _routing() -> RoutingConfig:
    return RoutingConfig(
        input_routing=[f"TAG_{i}" for i in range(6)],
        output_routing=["TAG_10", "TAG_11"],
        pids=[
            PIDConfig(
                pv_tag="TAG_1", cv_tag="TAG_2", setpoint=50.0, kp=1.0, ki=0.5, kd=0.1
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


def _unmapped_routing() -> RoutingConfig:
    return RoutingConfig(
        input_routing=["TAG_255"] * 6,
        output_routing=["TAG_255"] * 2,
        pids=[
            PIDConfig(
                pv_tag="TAG_255",
                cv_tag="TAG_255",
                setpoint=0.0,
                kp=0.0,
                ki=0.0,
                kd=0.0,
            )
            for _ in range(4)
        ],
        interlocks={
            f"TAG_{i}": InterlockConfig(
                lolo_limit=0.0,
                low_limit=5.0,
                high_limit=95.0,
                hihi_limit=100.0,
            )
            for i in range(32)
        },
    )


class TestReadTags:
    def test_decodes_32_floats(self) -> None:
        fake = _FakeModbusClient()
        regs: list[int] = []
        for i in range(32):
            regs.extend(float_to_registers(float(i)))
        fake.read_responses = [_FakeResponse(regs)]
        manager = _make_manager(fake)

        tags = asyncio.run(manager.read_tags())

        assert tags is not None
        assert len(tags) == 32
        assert tags["TAG_0"] == pytest.approx(0.0)
        assert tags["TAG_5"] == pytest.approx(5.0)
        assert tags["TAG_31"] == pytest.approx(31.0)

    def test_returns_none_when_disconnected(self) -> None:
        manager = _make_manager(_FakeModbusClient())
        manager._connected = False
        assert asyncio.run(manager.read_tags()) is None

    def test_marks_disconnected_on_read_error(self) -> None:
        fake = _FakeModbusClient()
        fake.force_read_error = True
        manager = _make_manager(fake)

        assert asyncio.run(manager.read_tags()) is None
        assert manager.connected is False


class TestReadRouting:
    def test_reassembles_routing_from_chunks(self) -> None:
        fake = _FakeModbusClient()
        # input(6), output(2), pid(40), then four 64-reg interlock chunks.
        pid_regs: list[int] = []
        for _ in range(4):
            pid_regs.extend([5, 6])  # pv=5, cv=6
            pid_regs.extend(float_to_registers(50.0))
            pid_regs.extend(float_to_registers(1.0))
            pid_regs.extend(float_to_registers(0.5))
            pid_regs.extend(float_to_registers(0.1))
        interlock_regs: list[int] = []
        for _ in range(32):
            interlock_regs.extend(float_to_registers(0.0))
            interlock_regs.extend(float_to_registers(5.0))
            interlock_regs.extend(float_to_registers(95.0))
            interlock_regs.extend(float_to_registers(100.0))
        fake.read_responses = [
            _FakeResponse([10, 11, 12, 13, 14, 15]),
            _FakeResponse([20, 21]),
            _FakeResponse(pid_regs),
            *[_FakeResponse(interlock_regs[o : o + 64]) for o in (0, 64, 128, 192)],
        ]
        manager = _make_manager(fake)

        config = asyncio.run(manager.read_routing())

        assert config is not None
        assert config.input_routing == [f"TAG_{i}" for i in range(10, 16)]
        assert config.output_routing == ["TAG_20", "TAG_21"]
        assert len(config.pids) == 4
        assert config.pids[0].pv_tag == "TAG_5"
        assert config.pids[0].setpoint == pytest.approx(50.0)
        assert len(config.interlocks) == 32
        assert config.interlocks["TAG_0"].hihi_limit == pytest.approx(100.0)

    def test_returns_none_on_pid_read_error(self) -> None:
        fake = _FakeModbusClient()
        fake.read_responses = [
            _FakeResponse([10, 11, 12, 13, 14, 15]),
            _FakeResponse([20, 21]),
            _FakeResponse(error=True),  # PID read fails
        ]
        manager = _make_manager(fake)
        assert asyncio.run(manager.read_routing()) is None


class TestWriteRouting:
    def test_writes_all_blocks_and_chunks_interlocks(self) -> None:
        fake = _FakeModbusClient()
        manager = _make_manager(fake)

        ok = asyncio.run(manager.write_routing(_routing()))

        assert ok is True
        # input + output + pid + 4 interlock chunks = 7 register writes.
        assert len(fake.write_calls) == 7
        # Each interlock chunk is tag-aligned at 64 registers.
        interlock_writes = fake.write_calls[3:]
        assert all(len(call["values"]) == 64 for call in interlock_writes)

    def test_writes_all_unmapped_routing_sentinel(self) -> None:
        fake = _FakeModbusClient()
        manager = _make_manager(fake)

        ok = asyncio.run(manager.write_routing(_unmapped_routing()))

        assert ok is True
        assert fake.write_calls[0]["values"] == [255] * 6
        assert fake.write_calls[1]["values"] == [255] * 2
        assert fake.write_calls[2]["values"][:2] == [255, 255]

    def test_returns_false_on_write_error(self) -> None:
        fake = _FakeModbusClient()
        fake.force_write_error = True
        manager = _make_manager(fake)
        assert asyncio.run(manager.write_routing(_routing())) is False

    def test_refuses_routing_with_malformed_tag(self) -> None:
        # A malformed routing tag must not be silently mapped to TAG_0; the
        # ValueError from the strict parser is caught and the write refused.
        fake = _FakeModbusClient()
        manager = _make_manager(fake)
        bad = _routing()
        bad.input_routing = ["TAG_0", "garbage", "TAG_2", "TAG_3", "TAG_4", "TAG_5"]
        assert asyncio.run(manager.write_routing(bad)) is False


class TestWriteTagAndSetpoint:
    def test_write_tag_refuses_the_republished_broker_block(self) -> None:
        """``TAG_n`` resolves to n*2 — a register the firmware republishes.

        The firmware rewrites 0..63 from its broker every scan and never reads
        them back, so this write can never reach the plant. It used to return
        True and put a value on the wire anyway (issue #4015).
        """
        fake = _FakeModbusClient()
        manager = _make_manager(fake)

        with pytest.raises(NotImplementedError):
            asyncio.run(manager.write_tag("TAG_5", 42.5))

        assert fake.write_calls == []

    def test_write_tag_rejects_unknown_tag(self) -> None:
        fake = _FakeModbusClient()
        manager = _make_manager(fake)
        # No write should be attempted for an unresolvable tag.
        assert asyncio.run(manager.write_tag("TAG_999", 1.0)) is False
        assert fake.write_calls == []

    def test_write_pid_setpoint_targets_setpoint_register(self) -> None:
        fake = _FakeModbusClient()
        manager = _make_manager(fake)

        ok = asyncio.run(manager.write_pid_setpoint(0, 12.5))

        assert ok is True
        # PID0 setpoint pair lives at PID_CONFIG_BASE + 2 = 202.
        assert fake.write_calls[0]["address"] == 202

    def test_write_pid_setpoint_rejects_bad_index(self) -> None:
        fake = _FakeModbusClient()
        manager = _make_manager(fake)
        assert asyncio.run(manager.write_pid_setpoint(99, 1.0)) is False
        assert fake.write_calls == []
