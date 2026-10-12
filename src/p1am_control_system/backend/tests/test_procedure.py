"""Tests for the declarative procedure executor (F09, #5530)."""

from __future__ import annotations

import ast
import pathlib

import pytest
from procedure import (
    NotSimulatedDriverError,
    Procedure,
    ProcedureExecutor,
    ProcedureState,
    Step,
)


class SimulatedPLCClient:
    """Stand-in carrying the simulated driver's class name (no hardware import)."""


class RealPLCClient:
    pass


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _always() -> bool:
    return True


def _make(steps, client=None):
    actions: list[str] = []
    safe: list[int] = []
    clock = Clock()
    ex = ProcedureExecutor(
        Procedure("p", tuple(steps)),
        client=client if client is not None else SimulatedPLCClient(),
        perform_action=actions.append,
        safe_state=lambda: safe.append(1),
        clock=clock,
    )
    return ex, actions, safe, clock


def test_steps_run_in_order() -> None:
    ex, actions, safe, _ = _make(
        [Step("a", _always, "A", 1.0), Step("b", _always, "B", 1.0)]
    )
    ex.start()
    ex.tick()
    ex.tick()
    assert actions == ["A", "B"]
    assert ex.state is ProcedureState.COMPLETE
    assert ex.completed_steps == ("a", "b")
    assert safe == []


def test_waits_for_precondition() -> None:
    flag = {"ok": False}
    ex, actions, _, _ = _make([Step("a", lambda: flag["ok"], "A", 5.0)])
    ex.start()
    ex.tick()
    assert actions == [] and ex.state is ProcedureState.RUNNING
    flag["ok"] = True
    ex.tick()
    assert actions == ["A"]


def test_hold_pauses_and_resumes() -> None:
    ex, actions, _, _ = _make(
        [Step("a", _always, "A", 1.0), Step("b", _always, "B", 1.0)]
    )
    ex.start()
    ex.tick()
    ex.hold()
    assert ex.state is ProcedureState.HOLD
    ex.tick()
    assert actions == ["A"]
    ex.resume()
    ex.tick()
    assert actions == ["A", "B"]


def test_hold_time_does_not_count_toward_timeout() -> None:
    ex, _, safe, clock = _make([Step("a", lambda: False, "A", 10.0)])
    ex.start()
    clock.now = 5.0
    ex.hold()
    clock.now = 500.0
    ex.resume()
    ex.tick()
    assert ex.state is ProcedureState.RUNNING and safe == []
    clock.now = 505.0
    ex.tick()
    assert ex.state is ProcedureState.ABORTED and safe == [1]


def test_abort_invokes_safe_state_exactly_once() -> None:
    ex, actions, safe, _ = _make(
        [Step("a", _always, "A", 1.0), Step("b", _always, "B", 1.0)]
    )
    ex.start()
    ex.tick()
    ex.abort()
    ex.abort()
    ex.tick()
    assert ex.state is ProcedureState.ABORTED
    assert safe == [1]
    assert actions == ["A"]


def test_abort_from_hold() -> None:
    ex, _, safe, _ = _make([Step("a", _always, "A", 1.0)])
    ex.start()
    ex.hold()
    ex.abort()
    assert ex.state is ProcedureState.ABORTED and safe == [1]


def test_non_simulated_client_refused_and_nothing_runs() -> None:
    ex, actions, safe, _ = _make([Step("a", _always, "A", 1.0)], client=RealPLCClient())
    with pytest.raises(
        NotSimulatedDriverError, match="SimulatedPLCClient.*RealPLCClient"
    ):
        ex.start()
    ex.tick()
    assert actions == [] and safe == []
    assert ex.state is ProcedureState.IDLE


def test_subclass_of_simulated_driver_accepted() -> None:
    class Child(SimulatedPLCClient):
        pass

    ex, actions, _, _ = _make([Step("a", _always, "A", 1.0)], client=Child())
    ex.start()
    ex.tick()
    assert actions == ["A"]


def test_step_timeout_aborts_and_de_energizes() -> None:
    ex, actions, safe, clock = _make([Step("a", lambda: False, "A", 2.0)])
    ex.start()
    clock.now = 1.9
    ex.tick()
    assert ex.state is ProcedureState.RUNNING
    clock.now = 2.0
    ex.tick()
    assert ex.state is ProcedureState.ABORTED
    assert safe == [1] and actions == []


def test_action_failure_aborts_and_de_energizes() -> None:
    safe: list[int] = []

    def boom(_: str) -> None:
        raise OSError("io")

    ex = ProcedureExecutor(
        Procedure("p", (Step("a", _always, "A", 1.0),)),
        client=SimulatedPLCClient(),
        perform_action=boom,
        safe_state=lambda: safe.append(1),
        clock=Clock(),
    )
    ex.start()
    ex.tick()
    assert ex.state is ProcedureState.ABORTED and safe == [1]


def test_precondition_error_aborts() -> None:
    def bad() -> bool:
        raise ValueError("x")

    ex, _, safe, _ = _make([Step("a", bad, "A", 1.0)])
    ex.start()
    ex.tick()
    assert ex.state is ProcedureState.ABORTED and safe == [1]


def test_safe_state_failure_propagates_but_state_is_aborted() -> None:
    def bad_safe() -> None:
        raise RuntimeError("relay stuck")

    ex = ProcedureExecutor(
        Procedure("p", (Step("a", _always, "A", 1.0),)),
        client=SimulatedPLCClient(),
        perform_action=lambda _: None,
        safe_state=bad_safe,
        clock=Clock(),
    )
    ex.start()
    with pytest.raises(RuntimeError, match="relay stuck"):
        ex.abort()
    assert ex.state is ProcedureState.ABORTED


def test_invalid_transitions_raise() -> None:
    ex, _, _, _ = _make([Step("a", _always, "A", 1.0)])
    with pytest.raises(RuntimeError):
        ex.hold()
    with pytest.raises(RuntimeError):
        ex.resume()
    ex.start()
    with pytest.raises(RuntimeError):
        ex.start()


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_step_rejects_bad_timeout(timeout: float) -> None:
    with pytest.raises(ValueError):
        Step("a", _always, "A", timeout)


def test_contract_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Step("", _always, "A", 1.0)
    with pytest.raises(ValueError):
        Step("a", _always, "", 1.0)
    with pytest.raises(ValueError):
        Procedure("p", ())
    with pytest.raises(ValueError):
        Procedure("", (Step("a", _always, "A", 1.0),))


def test_module_imports_no_hardware_driver() -> None:
    src = pathlib.Path(__file__).resolve().parents[1] / "procedure.py"
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {
        "hardware",
        "simulator_client",
        "modbus_client",
        "plc_factory",
        "plc_interface",
    }
    assert not (imported & forbidden)
