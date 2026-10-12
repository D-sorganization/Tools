"""Declarative procedure model with HOLD/ABORT, gated to the simulated driver.

F09 slice (#5530, part of #4089). A :class:`Procedure` is an ordered tuple of
:class:`Step` values; a :class:`ProcedureExecutor` walks it one ``tick()`` at a
time and exposes ``hold()``/``resume()``/``abort()`` transitions.

Design-by-contract, all collaborators injected -- this module imports **no**
hardware driver, performs **no** I/O and never touches a PLC itself:

* ``perform_action(action_id)`` runs a step's action.
* ``safe_state()`` de-energizes. It is the only hardware-facing callback the
  executor invokes on ABORT or any step failure, exactly once.
* ``clock()`` is monotonic seconds (injectable for tests).

Safety: ``abort()`` and the failure path never wait on a precondition, a lock
or a timeout; ``safe_state()`` is invoked straight after the state latches.
Starting is a *precondition*, not a warning: unless the injected PLC client is
the simulated driver, :meth:`ProcedureExecutor.start` raises
:class:`NotSimulatedDriverError` and runs nothing.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final

from shared.python.compatibility import StrEnum

__all__ = [
    "NotSimulatedDriverError",
    "Procedure",
    "ProcedureExecutor",
    "ProcedureState",
    "Step",
]

logger = logging.getLogger(__name__)

SIMULATED_DRIVER_NAME: Final = "SimulatedPLCClient"


class ProcedureState(StrEnum, Enum):
    """Executor states (IDLE/RUNNING names mirror the safety state machine)."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    HOLD = "HOLD"
    ABORTED = "ABORTED"
    COMPLETE = "COMPLETE"


class NotSimulatedDriverError(RuntimeError):
    """Raised when a procedure is started against a non-simulated PLC client."""


@dataclass(frozen=True)
class Step:
    """One procedure step.

    The action runs once ``precondition()`` is true; if that does not happen
    within ``timeout_s`` seconds of the step becoming active (HOLD time is
    excluded) the procedure aborts.
    """

    name: str
    precondition: Callable[[], bool]
    action_id: str
    timeout_s: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Step.name must be non-empty")
        if not self.action_id:
            raise ValueError("Step.action_id must be non-empty")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError(
                f"Step.timeout_s must be finite and > 0, got {self.timeout_s!r}"
            )


@dataclass(frozen=True)
class Procedure:
    """An ordered, immutable sequence of steps."""

    name: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Procedure.name must be non-empty")
        if not self.steps:
            raise ValueError("Procedure must contain at least one step")


def _is_simulated_driver(client: object) -> bool:
    """True if ``client`` is, or subclasses, the simulated PLC driver.

    Matched by class name along the MRO so this module need not import the
    driver (and, transitively, any hardware module).
    """
    return any(cls.__name__ == SIMULATED_DRIVER_NAME for cls in type(client).__mro__)


class ProcedureExecutor:
    """Runs a :class:`Procedure` with RUN/HOLD/ABORT transitions."""

    def __init__(
        self,
        procedure: Procedure,
        *,
        client: object,
        perform_action: Callable[[str], None],
        safe_state: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._procedure = procedure
        self._client = client
        self._perform_action = perform_action
        self._safe_state = safe_state
        self._clock = clock
        self._state = ProcedureState.IDLE
        self._index = 0
        self._deadline = 0.0
        self._hold_started = 0.0
        self._completed: list[str] = []

    @property
    def state(self) -> ProcedureState:
        return self._state

    @property
    def completed_steps(self) -> tuple[str, ...]:
        return tuple(self._completed)

    def start(self) -> None:
        """IDLE -> RUNNING. Raises unless the client is the simulated driver."""
        if not _is_simulated_driver(self._client):
            raise NotSimulatedDriverError(
                f"procedures run only against {SIMULATED_DRIVER_NAME}; "
                f"got {type(self._client).__name__}"
            )
        if self._state is not ProcedureState.IDLE:
            raise RuntimeError(f"cannot start from {self._state}")
        self._state = ProcedureState.RUNNING
        self._begin_step()

    def hold(self) -> None:
        """RUNNING -> HOLD. The step timeout clock is paused."""
        if self._state is not ProcedureState.RUNNING:
            raise RuntimeError(f"cannot hold from {self._state}")
        self._hold_started = self._clock()
        self._state = ProcedureState.HOLD

    def resume(self) -> None:
        """HOLD -> RUNNING, extending the step deadline by the time held."""
        if self._state is not ProcedureState.HOLD:
            raise RuntimeError(f"cannot resume from {self._state}")
        self._deadline += self._clock() - self._hold_started
        self._state = ProcedureState.RUNNING

    def abort(self, reason: str = "operator abort") -> None:
        """RUNNING/HOLD -> ABORTED, de-energizing exactly once.

        A no-op in any other state, so ``safe_state()`` is never re-run.
        """
        if self._state not in (ProcedureState.RUNNING, ProcedureState.HOLD):
            return
        logger.warning("procedure %s aborted: %s", self._procedure.name, reason)
        self._state = ProcedureState.ABORTED
        self._safe_state()

    def tick(self) -> ProcedureState:
        """Advance at most one step. A no-op unless RUNNING."""
        if self._state is not ProcedureState.RUNNING:
            return self._state
        step = self._procedure.steps[self._index]
        try:
            ready = bool(step.precondition())
        except Exception:
            logger.exception("precondition raised in step %s", step.name)
            self.abort(f"precondition error in step {step.name}")
            return self._state
        if ready:
            self._run_action(step)
        elif self._clock() >= self._deadline:
            self.abort(f"step {step.name} timed out after {step.timeout_s}s")
        return self._state

    def _run_action(self, step: Step) -> None:
        try:
            self._perform_action(step.action_id)
        except Exception:
            logger.exception("action %s failed in step %s", step.action_id, step.name)
            self.abort(f"action failure in step {step.name}")
            return
        self._completed.append(step.name)
        self._index += 1
        if self._index >= len(self._procedure.steps):
            self._state = ProcedureState.COMPLETE
        else:
            self._begin_step()

    def _begin_step(self) -> None:
        self._deadline = self._clock() + self._procedure.steps[self._index].timeout_s
