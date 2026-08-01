"""Shutdown must leave the plant de-energized, and must finish in time (#4005).

``deploy/install-services.sh`` sets ``TimeoutStopSec=15``. The old teardown set
``shutdown_event`` and then awaited three background tasks that only check the
event AFTER a long ``asyncio.sleep`` — so systemd SIGKILLed the process before
``alicat_manager.stop()`` or ``plc_client.disconnect()`` ever ran, killing the
Modbus socket mid-transaction and SQLite mid-WAL while the heater relay stayed
closed. Nothing in the teardown de-energized anything.

These tests pin the two properties that matter on real hardware:

1. the outputs are driven safe BEFORE anything is joined or closed, on the
   exception path too;
2. the whole teardown completes inside a deadline shorter than
   ``TimeoutStopSec``, even when a background task refuses to stop.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Generator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("PLC_DRIVER", "modbus")
# No auth fixture here on purpose: this module drives the lifespan and the
# safe-state seams directly and issues no HTTP request, so it needs no
# credential posture. It deliberately does NOT set P1AM_DEV_NO_AUTH at import
# time either — that mutation would leak into every suite collected after this
# one and make THEIR auth posture collection-order dependent (#4061).

pytest.importorskip("sqlmodel")
pytest.importorskip("fastapi")

import hardware  # noqa: E402
import main as backend_main  # noqa: E402
import shutdown_safety  # noqa: E402


@pytest.fixture(autouse=True)
def restore_latches() -> Generator[None, None, None]:
    """Keep the shared control-context / service latches isolated."""
    backend_main.shutdown_event.clear()
    try:
        yield
    finally:
        backend_main.control_context.clear_estop()
        backend_main.power_supply_service.clear_estop()
        backend_main.temperature_service.clear_estop()
        backend_main.shutdown_event.clear()


class _SafeStateSpy:
    """Records the ordered plant-safing actions issued against the PLC."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def write_coil(self, address: int, value: bool) -> bool:
        self.calls.append(f"coil@{address}={value}")
        return True

    async def write_pid_setpoint(self, pid_index: int, value: float) -> bool:
        self.calls.append(f"pid{pid_index}={value}")
        return True

    async def trigger_estop(self) -> bool:
        self.calls.append("trigger_estop")
        return True

    async def disconnect(self) -> None:
        self.calls.append("disconnect")


_RELAY_OPEN = f"coil@{hardware.HEATER_RELAY_COIL}=False"


def _CONTROLLERS() -> tuple[object, ...]:  # noqa: N802 - reads as a constant
    """The latchables the app wires into the shutdown sequence."""
    return (
        backend_main.control_context,
        backend_main.power_supply_service,
        backend_main.temperature_service,
    )


async def _drive_safe() -> None:
    await shutdown_safety.drive_outputs_safe(
        plc=backend_main.plc_client,
        controllers=_CONTROLLERS(),
        log=backend_main.logger,
    )


@contextlib.contextmanager
def _patched_plc(spy: _SafeStateSpy, *, connected: bool = True) -> Iterator[None]:
    """Swap the module-level PLC client's write seams for the spy."""
    plc = backend_main.plc_client
    with (
        patch.object(type(plc), "connected", property(lambda _self: connected)),
        patch.object(plc, "write_coil", spy.write_coil),
        patch.object(plc, "write_pid_setpoint", spy.write_pid_setpoint),
        patch.object(plc, "trigger_estop", spy.trigger_estop),
        patch.object(plc, "disconnect", spy.disconnect),
    ):
        yield


@contextlib.contextmanager
def _patched_startup(body_task: object) -> Iterator[None]:
    """Neutralise lifespan startup I/O and swap in controllable background tasks."""
    with (
        patch.object(backend_main, "init_db", MagicMock()),
        patch.object(backend_main, "Session", MagicMock()),
        patch.object(backend_main, "load_tags_into_plc_clients", MagicMock()),
        patch.object(backend_main, "_restore_persisted_settings", MagicMock()),
        patch.object(backend_main, "modbus_connect_background", body_task),
        patch.object(backend_main, "poll_plc_loop", body_task),
        patch.object(
            backend_main,
            "historian_retention_loop",
            lambda **_kwargs: body_task(),  # type: ignore[operator]
        ),
        patch.object(backend_main.alicat_manager, "start", MagicMock()),
        patch.object(backend_main.alicat_manager, "stop", AsyncMock()),
    ):
        yield


async def _cooperative_task() -> None:
    await backend_main.shutdown_event.wait()


async def _wedged_task() -> None:
    await asyncio.Event().wait()  # never set — models the 300 s retention sleep


class TestDriveOutputsSafe:
    @pytest.mark.asyncio
    async def test_drives_every_output_to_its_safe_state(self) -> None:
        spy = _SafeStateSpy()
        with _patched_plc(spy):
            await _drive_safe()

        assert spy.calls == [
            _RELAY_OPEN,
            f"pid{hardware.POWER_SUPPLY_PID_INDEX}=0.0",
            "trigger_estop",
        ]

    @pytest.mark.asyncio
    async def test_latches_the_controllers_so_poll_cannot_re_energize(self) -> None:
        spy = _SafeStateSpy()
        with _patched_plc(spy):
            await _drive_safe()

        assert backend_main.control_context.e_stop_active is True

    @pytest.mark.asyncio
    async def test_unacknowledged_write_does_not_raise(self) -> None:
        """A failed de-energize is logged, never raised: keep safing the rest."""
        plc = backend_main.plc_client
        with (
            patch.object(type(plc), "connected", property(lambda _self: True)),
            patch.object(plc, "write_coil", AsyncMock(return_value=False)),
            patch.object(plc, "write_pid_setpoint", AsyncMock(return_value=False)),
            patch.object(plc, "trigger_estop", AsyncMock(side_effect=OSError("gone"))),
        ):
            await _drive_safe()

    @pytest.mark.asyncio
    async def test_disconnected_plc_still_latches_the_controllers(self) -> None:
        spy = _SafeStateSpy()
        with _patched_plc(spy, connected=False):
            await _drive_safe()

        assert spy.calls == []
        assert backend_main.control_context.e_stop_active is True


class TestLifespanTeardown:
    @pytest.mark.asyncio
    async def test_safe_state_runs_before_disconnect(self) -> None:
        spy = _SafeStateSpy()
        with _patched_startup(_cooperative_task), _patched_plc(spy):
            async with backend_main.lifespan(backend_main.app):
                pass

        assert spy.calls[-1] == "disconnect"
        assert spy.calls.index("trigger_estop") < spy.calls.index("disconnect")

    @pytest.mark.asyncio
    async def test_safe_state_runs_on_the_exception_path(self) -> None:
        """An unhandled error in the app body must still de-energize."""
        spy = _SafeStateSpy()
        propagated = False
        with _patched_startup(_cooperative_task), _patched_plc(spy):
            try:
                async with backend_main.lifespan(backend_main.app):
                    raise RuntimeError("boom")
            except RuntimeError:
                propagated = True

        assert propagated, "the app-body error must still propagate"
        assert _RELAY_OPEN in spy.calls
        assert "disconnect" in spy.calls

    @pytest.mark.asyncio
    async def test_wedged_task_cannot_starve_the_shutdown(self) -> None:
        """A task that ignores shutdown_event must not eat TimeoutStopSec."""
        spy = _SafeStateSpy()
        loop = asyncio.get_running_loop()
        with (
            _patched_startup(_wedged_task),
            _patched_plc(spy),
            patch.object(shutdown_safety, "SHUTDOWN_DEADLINE_S", 0.3),
        ):
            started = loop.time()
            async with backend_main.lifespan(backend_main.app):
                pass
            elapsed = loop.time() - started

        assert elapsed < 5.0, f"teardown took {elapsed:.1f}s — SIGKILL territory"
        # The safety-critical writes and closes still ran despite the wedge.
        assert _RELAY_OPEN in spy.calls
        assert "disconnect" in spy.calls

    @pytest.mark.asyncio
    async def test_hung_safe_state_write_still_lets_the_closes_run(self) -> None:
        """A half-open Modbus socket can leave a write pending forever.

        The safe-state attempt is bounded so it cannot starve the closes it is
        meant to precede.
        """

        async def _never_returns(*_args: object, **_kwargs: object) -> bool:
            await asyncio.Event().wait()
            return True  # pragma: no cover

        spy = _SafeStateSpy()
        plc = backend_main.plc_client
        loop = asyncio.get_running_loop()
        with (
            _patched_startup(_cooperative_task),
            patch.object(type(plc), "connected", property(lambda _self: True)),
            patch.object(plc, "write_coil", _never_returns),
            patch.object(plc, "disconnect", spy.disconnect),
            patch.object(shutdown_safety, "SAFE_STATE_DEADLINE_S", 0.2),
        ):
            started = loop.time()
            async with backend_main.lifespan(backend_main.app):
                pass
            elapsed = loop.time() - started

        assert elapsed < 5.0, f"teardown took {elapsed:.1f}s — SIGKILL territory"
        assert "disconnect" in spy.calls

    def test_shutdown_deadline_is_below_systemd_timeout_stop_sec(self) -> None:
        """deploy/install-services.sh sets TimeoutStopSec=15."""
        assert shutdown_safety.SHUTDOWN_DEADLINE_S < 15.0
        assert (
            shutdown_safety.SAFE_STATE_DEADLINE_S < shutdown_safety.SHUTDOWN_DEADLINE_S
        )


class TestConnectLoopShutdownLatency:
    @pytest.mark.asyncio
    async def test_connect_loop_wakes_immediately_on_shutdown(self) -> None:
        """The retry sleep must be interruptible, not a fixed blocking sleep."""
        backend_main.shutdown_event.clear()
        with (
            patch.object(backend_main, "_connect_once", AsyncMock(return_value=None)),
            patch.object(backend_main.settings, "connect_retry_interval_s", 30.0),
        ):
            task = asyncio.create_task(backend_main.modbus_connect_background())
            await asyncio.sleep(0.05)
            backend_main.shutdown_event.set()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except TimeoutError:  # pragma: no cover - failure path
                task.cancel()
                pytest.fail(
                    "connect loop slept through shutdown_event; it must wait on "
                    "the event, not on a bare asyncio.sleep"
                )
