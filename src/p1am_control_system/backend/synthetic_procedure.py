"""Bounded simulator-only procedure state machine with attributable events."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

from identity import Principal, Role
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    # Type checkers must see the real 3.11 symbol; TYPE_CHECKING is always
    # true for them and always false at runtime, so this needs no version
    # test and never degrades StrEnum members to bare `str`.
    from enum import StrEnum
else:
    from enum_compat import StrEnum


class ProcedureState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    HOLDING = "holding"
    STOPPING = "stopping"
    ABORTED = "aborted"
    RECOVERING = "recovering"


class ProcedureCommand(StrEnum):
    START = "start"
    RUN = "run"
    HOLD = "hold"
    RESUME = "resume"
    STOP = "stop"
    COMPLETE = "complete"
    ABORT = "abort"
    RECOVER = "recover"
    TIMEOUT = "timeout"


class ProcedureEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence: int
    command: ProcedureCommand
    before: ProcedureState
    after: ProcedureState
    actor: str
    reason: str
    occurred_at: datetime
    deadline: datetime | None
    data_classification: Literal["synthetic"] = "synthetic"
    not_for_live_control: Literal[True] = True


_TRANSITIONS = {
    (ProcedureState.IDLE, ProcedureCommand.START): ProcedureState.STARTING,
    (ProcedureState.STARTING, ProcedureCommand.RUN): ProcedureState.RUNNING,
    (ProcedureState.RUNNING, ProcedureCommand.HOLD): ProcedureState.HOLDING,
    (ProcedureState.HOLDING, ProcedureCommand.RESUME): ProcedureState.RUNNING,
    (ProcedureState.RUNNING, ProcedureCommand.STOP): ProcedureState.STOPPING,
    (ProcedureState.HOLDING, ProcedureCommand.STOP): ProcedureState.STOPPING,
    (ProcedureState.STOPPING, ProcedureCommand.COMPLETE): ProcedureState.IDLE,
    (ProcedureState.ABORTED, ProcedureCommand.RECOVER): ProcedureState.RECOVERING,
    (ProcedureState.RECOVERING, ProcedureCommand.COMPLETE): ProcedureState.IDLE,
}
_BOUNDED_STATES = {
    ProcedureState.STARTING,
    ProcedureState.STOPPING,
    ProcedureState.RECOVERING,
}


class SyntheticProcedure:
    def __init__(
        self,
        now: Callable[[], datetime],
        transition_timeout: timedelta = timedelta(minutes=2),
    ) -> None:
        if transition_timeout <= timedelta(0):
            raise ValueError("transition_timeout must be positive")
        self._now = now
        self._timeout = transition_timeout
        self._state = ProcedureState.IDLE
        self._deadline: datetime | None = None
        self._events: list[ProcedureEvent] = []

    @property
    def state(self) -> ProcedureState:
        return self._state

    def events(self) -> list[ProcedureEvent]:
        return list(self._events)

    def _record(
        self,
        command: ProcedureCommand,
        after: ProcedureState,
        actor: str,
        reason: str,
    ) -> ProcedureEvent:
        occurred_at = self._now()
        before = self._state
        self._state = after
        self._deadline = (
            occurred_at + self._timeout if after in _BOUNDED_STATES else None
        )
        event = ProcedureEvent(
            sequence=len(self._events) + 1,
            command=command,
            before=before,
            after=after,
            actor=actor,
            reason=reason.strip(),
            occurred_at=occurred_at,
            deadline=self._deadline,
        )
        self._events.append(event)
        return event

    def dispatch(
        self,
        command: ProcedureCommand,
        principal: Principal,
        reason: str,
    ) -> ProcedureEvent:
        if principal.role is Role.VIEWER:
            raise PermissionError("operator, engineer, or admin role required")
        if not reason.strip():
            raise ValueError("transition reason is required")
        if command is ProcedureCommand.ABORT and self._state is not ProcedureState.IDLE:
            after = ProcedureState.ABORTED
        else:
            try:
                after = _TRANSITIONS[(self._state, command)]
            except KeyError as exc:
                raise ValueError(
                    f"{command.value} is not allowed from {self._state.value}"
                ) from exc
        return self._record(command, after, principal.subject, reason)

    def enforce_deadline(self) -> ProcedureEvent | None:
        if self._deadline is None or self._now() <= self._deadline:
            return None
        return self._record(
            ProcedureCommand.TIMEOUT,
            ProcedureState.ABORTED,
            "synthetic.procedure.supervisor",
            "Bounded transition deadline exceeded",
        )
