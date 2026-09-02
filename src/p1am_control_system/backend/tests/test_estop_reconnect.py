"""E-stop reconnect + write-seam interlock defense-in-depth tests.

Covers the hardening added on the heater / power-supply write path:

    * ``connect_once`` / ``poll_once`` re-engage the process-local controller
      latches (and arm the write-seam interlocks) while ``estop_active`` is set,
      so a PLC reconnect can never let the next scan re-energize an output
      before the hardware E-stop is re-asserted.
    * The low-level and service write seams honor the shared E-stop flag
      independently of the controller: an *energizing* write is forced to the
      safe OFF/0 direction; the *de-energizing* direction is never blocked.
    * A failed de-energize (commanded OFF/0 but the write raises/errors) retries
      once and surfaces a comms failure rather than silently holding the last
      state.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import hardware  # noqa: E402
from poll_runtime import connect_once, poll_once  # noqa: E402
from power_supply_integration import PowerSupplyService  # noqa: E402
from temperature_integration import TemperatureService  # noqa: E402

# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class _RecordingService:
    """Records engage_estop / set_estop_active calls; mimics the real seam."""

    def __init__(self) -> None:
        self.engage_calls = 0
        self.estop_flag: bool | None = None
        self.seen_tags: list[Any] = []
        self.controller = type(
            "Controller",
            (),
            {"config": type("Config", (), {"command_tag": "TAG_10"})()},
        )()

    def engage_estop(self) -> None:
        self.engage_calls += 1

    def set_estop_active(self, active: bool) -> None:
        self.estop_flag = active

    async def poll(self, tags: Any) -> Any:
        self.seen_tags.append(tags)
        return _Status()


class _Status:
    def model_dump(self) -> dict[str, str]:
        return {"state": "ok"}


class _ConnectPLC:
    def __init__(self) -> None:
        self.connected = False
        self.trigger_count = 0
        self.estop_flag: bool | None = None

    async def connect(self) -> bool:
        self.connected = True
        return True

    def set_estop_active(self, active: bool) -> None:
        self.estop_flag = active

    async def trigger_estop(self) -> None:
        self.trigger_count += 1

    async def read_routing(self) -> None:
        return None


class _PollPLC:
    def __init__(self, tags: dict[str, float] | None) -> None:
        self.connected = True
        self._tags = tags
        self.trigger_count = 0
        self.estop_flag: bool | None = None

    async def read_tags(self) -> dict[str, float] | None:
        return self._tags

    def set_estop_active(self, active: bool) -> None:
        self.estop_flag = active

    async def trigger_estop(self) -> None:
        self.trigger_count += 1


class _Sim:
    async def read_tags(self) -> dict[str, float] | None:
        return None


class _Ws:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def broadcast(self, message: Any) -> None:
        self.messages.append(message)


class _Alicats:
    def get_devices_data(self) -> list[dict[str, str]]:
        return []


class _AlarmEngine:
    def update_tag(self, name: str, value: float) -> list[dict[str, str]]:
        return []


class _NullHistorian:
    """Historian sink double: the scan queues rows, it never writes them here."""

    def __init__(self) -> None:
        self.records: list[Any] = []

    def submit(self, record: Any) -> bool:
        self.records.append(record)
        return True


# --------------------------------------------------------------------------
# H4 — reconnect / poll re-engages the controller latches while estopped
# --------------------------------------------------------------------------


class TestReconnectReengagesLatches:
    def test_connect_once_reengages_service_latches_when_estopped(self) -> None:
        async def _go() -> None:
            plc = _ConnectPLC()
            power = _RecordingService()
            temp = _RecordingService()
            await connect_once(
                plc=plc,
                power_supply=power,
                apply_config=lambda _c: None,
                estop_active=True,
                temperature=temp,
            )
            assert power.engage_calls == 1
            assert temp.engage_calls == 1
            assert power.estop_flag is True
            assert temp.estop_flag is True
            assert plc.estop_flag is True
            assert plc.trigger_count == 1

        asyncio.run(_go())

    def test_connect_once_does_not_reengage_when_not_estopped(self) -> None:
        async def _go() -> None:
            plc = _ConnectPLC()
            power = _RecordingService()
            temp = _RecordingService()
            await connect_once(
                plc=plc,
                power_supply=power,
                apply_config=lambda _c: None,
                estop_active=False,
                temperature=temp,
            )
            assert power.engage_calls == 0
            assert temp.engage_calls == 0
            assert plc.trigger_count == 0

        asyncio.run(_go())

    def test_poll_once_reengages_latches_before_polls_when_estopped(self) -> None:
        async def _go() -> None:
            plc = _PollPLC({"TAG_0": 1.0})
            power = _RecordingService()
            temp = _RecordingService()
            await poll_once(
                plc=plc,
                backup=_Sim(),
                latest_tag_values={"TAG_0": 0.0},
                ws=_Ws(),
                alicats=_Alicats(),
                power_supply=power,
                temperature=temp,
                alarm_engine=_AlarmEngine(),
                active_alarm_map={},
                estop_active=True,
            )
            # Latches re-engaged, write-seam interlocks armed, and the polls ran.
            assert power.engage_calls == 1
            assert temp.engage_calls == 1
            assert power.estop_flag is True
            assert temp.estop_flag is True
            assert plc.estop_flag is True
            assert power.seen_tags and temp.seen_tags
            assert plc.trigger_count == 1

        asyncio.run(_go())

    def test_poll_once_lowers_write_seam_flag_when_not_estopped(self) -> None:
        async def _go() -> None:
            plc = _PollPLC({"TAG_0": 1.0})
            power = _RecordingService()
            temp = _RecordingService()
            await poll_once(
                plc=plc,
                backup=_Sim(),
                latest_tag_values={"TAG_0": 0.0},
                ws=_Ws(),
                alicats=_Alicats(),
                power_supply=power,
                temperature=temp,
                alarm_engine=_AlarmEngine(),
                active_alarm_map={},
                estop_active=False,
            )
            # Controller latch is NOT auto-engaged, but the low-level interlock
            # flag tracks the shared flag down to False.
            assert power.engage_calls == 0
            assert power.estop_flag is False
            assert temp.estop_flag is False
            assert plc.estop_flag is False
            assert plc.trigger_count == 0

        asyncio.run(_go())


# --------------------------------------------------------------------------
# H5 — write-seam interlock forces energize -> safe when the flag is set
# --------------------------------------------------------------------------


class TestServiceWriteSeamInterlock:
    def test_temperature_relay_energize_forced_off_when_flag_set(self) -> None:
        async def _go() -> None:
            plc = MagicMock()
            plc.write_coil = AsyncMock(return_value=True)
            svc = TemperatureService(plc, logging.getLogger("test"))
            svc.set_estop_active(True)
            ok = await svc._write_relay(True)  # commanded ON
            assert ok is True
            # Interlock forced it OFF at the seam, independent of the controller.
            plc.write_coil.assert_awaited_once_with(hardware.HEATER_RELAY_COIL, False)

        asyncio.run(_go())

    def test_temperature_deenergize_not_blocked_by_flag(self) -> None:
        async def _go() -> None:
            plc = MagicMock()
            plc.write_coil = AsyncMock(return_value=True)
            svc = TemperatureService(plc, logging.getLogger("test"))
            svc.set_estop_active(True)
            ok = await svc._write_relay(False)
            assert ok is True
            plc.write_coil.assert_awaited_once_with(hardware.HEATER_RELAY_COIL, False)

        asyncio.run(_go())

    def test_power_supply_energize_forced_zero_when_flag_set(self) -> None:
        async def _go() -> None:
            plc = MagicMock()
            plc.write_pid_setpoint = AsyncMock(return_value=True)
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            svc.set_estop_active(True)
            ok = await svc._write_pid_setpoint(0, 42.0)  # energizing command
            assert ok is True
            plc.write_pid_setpoint.assert_awaited_once_with(0, 0.0)

        asyncio.run(_go())

    def test_modbus_write_coil_interlock_forces_energize_off(self) -> None:
        async def _go() -> None:
            mgr = _modbus_manager()
            mgr.set_estop_active(True)
            ok = await mgr.write_coil(hardware.HEATER_RELAY_COIL, True)
            assert ok is True
            mgr.client.write_coil.assert_awaited_once_with(
                address=hardware.HEATER_RELAY_COIL, value=False
            )

        asyncio.run(_go())

    def test_modbus_write_pid_setpoint_interlock_forces_zero(self) -> None:
        async def _go() -> None:
            mgr = _modbus_manager()
            mgr.set_estop_active(True)
            ok = await mgr.write_pid_setpoint(0, 55.0)
            assert ok is True
            # The register values written must encode 0.0, not the commanded 55.
            from modbus_codec import float_to_registers

            mgr.client.write_registers.assert_awaited_once()
            written = mgr.client.write_registers.await_args.kwargs["values"]
            assert written == float_to_registers(0.0)

        asyncio.run(_go())

    def test_set_estop_active_rejects_non_bool(self) -> None:
        svc = TemperatureService(MagicMock(), logging.getLogger("test"))
        with pytest.raises(TypeError):
            svc.set_estop_active(1)
        mgr = _modbus_manager()
        with pytest.raises(TypeError):
            mgr.set_estop_active("yes")


# --------------------------------------------------------------------------
# H6 — a failed de-energize retries once and signals comms failure
# --------------------------------------------------------------------------


class TestDeenergizeRetryAndCommsFailure:
    def test_temperature_deenergize_retries_and_flags_comms_failure(self) -> None:
        async def _go() -> None:
            plc = MagicMock()
            plc.write_coil = AsyncMock(side_effect=RuntimeError("bus dropped"))
            svc = TemperatureService(plc, logging.getLogger("test"))
            ok = await svc._write_relay(False)  # commanded OFF, keeps failing
            assert ok is False
            # Retried once -> two attempts total.
            assert plc.write_coil.await_count == 2
            assert svc.deenergize_comms_failed is True

        asyncio.run(_go())

    def test_temperature_deenergize_success_clears_comms_flag(self) -> None:
        async def _go() -> None:
            plc = MagicMock()
            # Fail once, then succeed on the retry.
            plc.write_coil = AsyncMock(side_effect=[RuntimeError("blip"), True])
            svc = TemperatureService(plc, logging.getLogger("test"))
            ok = await svc._write_relay(False)
            assert ok is True
            assert plc.write_coil.await_count == 2
            assert svc.deenergize_comms_failed is False

        asyncio.run(_go())

    def test_power_supply_deenergize_retries_and_flags_comms_failure(self) -> None:
        async def _go() -> None:
            plc = MagicMock()
            plc.write_pid_setpoint = AsyncMock(return_value=False)
            svc = PowerSupplyService(plc, logging.getLogger("test"))
            ok = await svc._write_pid_setpoint(0, 0.0)  # de-energize, keeps failing
            assert ok is False
            assert plc.write_pid_setpoint.await_count == 2
            assert svc.deenergize_comms_failed is True

        asyncio.run(_go())

    def test_energize_write_is_not_retried(self) -> None:
        async def _go() -> None:
            plc = MagicMock()
            plc.write_coil = AsyncMock(return_value=False)
            svc = TemperatureService(plc, logging.getLogger("test"))
            ok = await svc._write_relay(True)  # energizing; single attempt only
            assert ok is False
            assert plc.write_coil.await_count == 1
            # An energize failure does not raise a de-energize comms alarm.
            assert svc.deenergize_comms_failed is False

        asyncio.run(_go())

    def test_modbus_write_coil_deenergize_retries_then_fails(self) -> None:
        async def _go() -> None:
            mgr = _modbus_manager()
            mgr.client.write_coil = AsyncMock(
                return_value=MagicMock(isError=lambda: True)
            )
            ok = await mgr.write_coil(hardware.HEATER_RELAY_COIL, False)
            assert ok is False
            assert mgr.client.write_coil.await_count == 2

        asyncio.run(_go())

    def test_modbus_write_pid_setpoint_deenergize_retries_then_fails(self) -> None:
        async def _go() -> None:
            mgr = _modbus_manager()
            mgr.client.write_registers = AsyncMock(
                return_value=MagicMock(isError=lambda: True)
            )
            ok = await mgr.write_pid_setpoint(0, 0.0)
            assert ok is False
            assert mgr.client.write_registers.await_count == 2

        asyncio.run(_go())


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _modbus_manager() -> Any:
    """A connected AsyncModbusManager wired to a MagicMock pymodbus client."""
    from modbus_client import AsyncModbusManager

    mgr = AsyncModbusManager(host="127.0.0.1")
    mgr.connected = True
    client = MagicMock()
    client.write_coil = AsyncMock(return_value=MagicMock(isError=lambda: False))
    client.write_registers = AsyncMock(return_value=MagicMock(isError=lambda: False))
    mgr.client = client
    return mgr
