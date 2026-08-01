"""Ordered, deadline-bounded teardown that leaves the plant de-energized.

Stopping this process is a plant operation, not just a resource cleanup, so the
sequencing lives here rather than inline in the FastAPI app module.

The old teardown set the shutdown flag, awaited three background tasks that only
checked the flag AFTER a long ``asyncio.sleep``, and only then closed the Modbus
socket and the historian. With ``TimeoutStopSec=15`` in
``deploy/install-services.sh`` that meant systemd SIGKILLed the process before
either close ran — the Modbus socket died mid-transaction and SQLite was killed
mid-WAL — all while the heater relay stayed closed, because nothing in the
teardown de-energized anything (issue #4005).

Two invariants follow, and both are enforced by tests:

* the plant is driven safe BEFORE anything is joined or closed, on the exception
  path as well as a clean stop;
* the whole sequence finishes inside a deadline shorter than
  ``TimeoutStopSec``, even when a background task refuses to stop or a
  half-open socket leaves a write pending forever.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from logging import Logger
from typing import Any, Protocol

import hardware

# Total wall-clock budget for the teardown. MUST stay comfortably below the
# unit's TimeoutStopSec. Anything that overruns its slice is cancelled, not
# waited on — an orderly close of the remaining resources is worth more than a
# clean exit of one wedged task.
SHUTDOWN_DEADLINE_S = 10.0
# Slice of that budget reserved for driving the plant safe. It runs first and is
# itself bounded: a safe-state attempt that hangs on a half-open socket would
# starve the closes it is meant to precede. On expiry the firmware watchdog
# (hardware.HEARTBEAT_TIMEOUT_S with no heartbeat) is the remaining backstop.
SAFE_STATE_DEADLINE_S = 4.0
# Floor for each step so none gets a zero/negative timeout after earlier steps
# consumed the budget.
_STEP_MIN_S = 0.5


class SafeablePLC(Protocol):
    """The subset of the PLC client the safe-state drive needs."""

    @property
    def connected(self) -> bool: ...

    async def write_coil(self, address: int, value: bool) -> bool: ...

    async def write_pid_setpoint(self, pid_index: int, value: float) -> bool: ...

    async def trigger_estop(self) -> bool: ...


class Latchable(Protocol):
    """A controller or context that can latch itself into the tripped state."""

    def engage_estop(self) -> None: ...


async def verified_safe_write(label: str, write: Awaitable[bool], log: Logger) -> None:
    """Await a de-energizing write and escalate loudly if it is not acked.

    Never raises: the process is exiting, so one refused write must not skip the
    writes that follow it. An unacknowledged de-energize is logged at CRITICAL
    because nothing downstream will retry it.
    """
    try:
        acknowledged = bool(await write)
    except Exception as exc:  # noqa: BLE001 - shutdown must continue regardless
        log.critical("Shutdown safe-state: %s raised (%s)", label, exc)
        return
    if acknowledged:
        log.info("Shutdown safe-state: %s confirmed.", label)
    else:
        log.critical("Shutdown safe-state: %s NOT acknowledged by the PLC.", label)


async def drive_outputs_safe(
    *, plc: SafeablePLC, controllers: Sequence[Latchable], log: Logger
) -> None:
    """Drive every physical output to its de-energized state.

    Order is a safety property:

    1. the controller latches go up so a still-running poll loop cannot
       re-command an output behind us (same ordering as ``POST /api/estop``);
    2. the heater relay opens — the only thing commanding the 110 V element;
    3. the power-supply analog command is zeroed;
    4. the PLC-side E-stop is asserted last, latching over an already-safe plant
       rather than racing it.

    Every write is verified individually. Best-effort by design: a refused write
    is escalated but never aborts the remaining steps.
    """
    for controller in controllers:
        controller.engage_estop()
    # Raise the client's own write-seam latch so anything still in flight is
    # forced to the safe direction (defense in depth; optional on the base API).
    set_plc_latch = getattr(plc, "set_estop_active", None)
    if callable(set_plc_latch):
        set_plc_latch(True)

    if not plc.connected:
        log.warning(
            "Shutdown safe-state: PLC not connected — controllers latched, but "
            "outputs could NOT be commanded safe from here. The firmware "
            "watchdog (%.1fs without a heartbeat) is the remaining protection.",
            hardware.HEARTBEAT_TIMEOUT_S,
        )
        return

    await verified_safe_write(
        "heater relay opened",
        plc.write_coil(hardware.HEATER_RELAY_COIL, False),
        log,
    )
    await verified_safe_write(
        "power-supply setpoint zeroed",
        plc.write_pid_setpoint(hardware.POWER_SUPPLY_PID_INDEX, 0.0),
        log,
    )
    await verified_safe_write("PLC E-stop asserted", plc.trigger_estop(), log)


def step_budget(deadline: float) -> float:
    """Seconds left before ``deadline``, floored so a step always gets a chance."""
    return max(_STEP_MIN_S, deadline - time.monotonic())


async def guarded_step(
    label: str, step: Awaitable[Any], budget: float, log: Logger
) -> None:
    """Await a teardown step under a timeout; log and continue on failure."""
    try:
        await asyncio.wait_for(step, timeout=budget)
    except TimeoutError:
        log.error("Shutdown: %s exceeded its %.1fs budget.", label, budget)
    except Exception as exc:  # noqa: BLE001 - one failed close must not skip the rest
        log.error("Shutdown: %s failed: %s", label, exc)


async def stop_background_tasks(
    tasks: Sequence[asyncio.Task[Any]], budget: float, log: Logger
) -> None:
    """Join the background tasks, cancelling any that overrun the budget."""
    if not tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=budget
        )
        return
    except TimeoutError:
        log.error(
            "Shutdown: background tasks did not stop within %.1fs; cancelling "
            "so the Modbus/SQLite closes still run before SIGKILL.",
            budget,
        )
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def run_shutdown_sequence(
    *,
    plc: SafeablePLC,
    controllers: Sequence[Latchable],
    shutdown_event: asyncio.Event,
    tasks: Sequence[asyncio.Task[Any]],
    closers: Sequence[tuple[str, Callable[[], Awaitable[Any]]]],
    log: Logger,
) -> None:
    """Safe the plant, stop the background tasks, then close I/O — all bounded.

    Args:
        plc: The active PLC client (its write seams drive the outputs safe).
        controllers: Objects latched into the tripped state first, so nothing
            can re-command an output while the safing writes are in flight.
        shutdown_event: Set only AFTER the plant is safe, so the loops keep
            running (and re-asserting) until then.
        tasks: Background tasks to join, cancelled if they overrun.
        closers: ``(label, factory)`` pairs; each factory is called to build the
            close coroutine so nothing is created until its turn comes.
        log: Logger for the escalation trail.
    """
    deadline = time.monotonic() + SHUTDOWN_DEADLINE_S
    await guarded_step(
        "plant safe-state",
        drive_outputs_safe(plc=plc, controllers=controllers, log=log),
        min(SAFE_STATE_DEADLINE_S, step_budget(deadline)),
        log,
    )
    shutdown_event.set()
    await stop_background_tasks(tasks, step_budget(deadline), log)
    for label, factory in closers:
        await guarded_step(label, factory(), step_budget(deadline), log)
