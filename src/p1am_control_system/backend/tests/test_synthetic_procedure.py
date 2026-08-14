"""F09 deterministic simulator-only procedure contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

import pytest
from identity import Principal, Role
from synthetic_procedure import ProcedureCommand, ProcedureState, SyntheticProcedure


def _principal() -> Principal:
    return Principal("operator.one", "Operator One", Role.OPERATOR)


def test_start_run_hold_resume_stop_cycle_is_deterministic_and_attributed() -> None:
    clock = [datetime(2026, 8, 3, 20, 0, tzinfo=UTC)]
    procedure = SyntheticProcedure(now=lambda: clock[0])

    events = [
        procedure.dispatch(ProcedureCommand.START, _principal(), "Begin synthetic run"),
        procedure.dispatch(ProcedureCommand.RUN, _principal(), "Start checks complete"),
        procedure.dispatch(ProcedureCommand.HOLD, _principal(), "Synthetic hold"),
        procedure.dispatch(
            ProcedureCommand.RESUME, _principal(), "Resume synthetic run"
        ),
        procedure.dispatch(ProcedureCommand.STOP, _principal(), "Normal stop"),
        procedure.dispatch(
            ProcedureCommand.COMPLETE, _principal(), "Stop checks complete"
        ),
    ]

    assert [event.after for event in events] == [
        ProcedureState.STARTING,
        ProcedureState.RUNNING,
        ProcedureState.HOLDING,
        ProcedureState.RUNNING,
        ProcedureState.STOPPING,
        ProcedureState.IDLE,
    ]
    assert all(event.actor == "operator.one" for event in events)
    assert all(event.data_classification == "synthetic" for event in events)
    assert procedure.state is ProcedureState.IDLE


def test_abort_and_recovery_are_bounded() -> None:
    clock = [datetime(2026, 8, 3, 20, 0, tzinfo=UTC)]
    procedure = SyntheticProcedure(
        now=lambda: clock[0],
        transition_timeout=timedelta(seconds=30),
    )
    procedure.dispatch(ProcedureCommand.START, _principal(), "Begin synthetic run")
    abort = procedure.dispatch(ProcedureCommand.ABORT, _principal(), "Synthetic fault")
    recovery = procedure.dispatch(
        ProcedureCommand.RECOVER, _principal(), "Recovery approved"
    )

    assert abort.after is ProcedureState.ABORTED
    assert recovery.after is ProcedureState.RECOVERING
    assert recovery.deadline == clock[0] + timedelta(seconds=30)

    clock[0] += timedelta(seconds=31)
    timeout = procedure.enforce_deadline()
    assert timeout is not None
    assert timeout.after is ProcedureState.ABORTED
    assert timeout.command is ProcedureCommand.TIMEOUT


def test_invalid_transition_is_fail_closed() -> None:
    procedure = SyntheticProcedure(now=lambda: datetime(2026, 8, 3, tzinfo=UTC))

    with pytest.raises(ValueError, match="not allowed"):
        procedure.dispatch(ProcedureCommand.RUN, _principal(), "Invalid direct run")
