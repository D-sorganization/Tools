"""Tests for data_processor.core.undo_redo module."""

from __future__ import annotations

import pandas as pd
import pytest

from data_processor.core.undo_redo import (
    Command,
    CommandRecord,
    UndoRedoManager,
)


# ─── Concrete command for testing ──────────────────────────────────────

class IncrementCommand(Command):
    """Test command that increments a counter."""

    def __init__(self, counter: list[int], amount: int = 1) -> None:
        self._counter = counter
        self._amount = amount

    @property
    def name(self) -> str:
        return "increment"

    @property
    def description(self) -> str:
        return f"Increment by {self._amount}"

    def execute(self) -> int:
        self._counter[0] += self._amount
        return self._counter[0]

    def undo(self) -> int:
        self._counter[0] -= self._amount
        return self._counter[0]


class FailingCommand(Command):
    """A command whose execute always raises."""

    @property
    def name(self) -> str:
        return "fail"

    @property
    def description(self) -> str:
        return "Always fails"

    def execute(self) -> None:
        raise RuntimeError("boom")

    def undo(self) -> None:
        pass


# ─── Tests ─────────────────────────────────────────────────────────────


class TestCommandInterface:
    """Tests for the abstract Command base class."""

    def test_redo_defaults_to_execute(self) -> None:
        counter = [0]
        cmd = IncrementCommand(counter, amount=5)
        cmd.execute()
        assert counter[0] == 5
        cmd.undo()
        assert counter[0] == 0
        cmd.redo()
        assert counter[0] == 5

    def test_name_and_description(self) -> None:
        cmd = IncrementCommand([0], 3)
        assert cmd.name == "increment"
        assert "3" in cmd.description


class TestCommandRecord:
    """Tests for CommandRecord dataclass."""

    def test_record_creation(self) -> None:
        cmd = IncrementCommand([0])
        record = CommandRecord(command=cmd, result=42)
        assert record.command is cmd
        assert record.result == 42
        assert record.executed_at is not None


class TestUndoRedoManager:
    """Tests for UndoRedoManager class."""

    def test_initial_state(self) -> None:
        mgr = UndoRedoManager()
        assert not mgr.can_undo
        assert not mgr.can_redo
        assert mgr.history == []

    def test_execute_adds_to_history(self) -> None:
        mgr = UndoRedoManager()
        counter = [0]
        mgr.execute(IncrementCommand(counter))
        assert counter[0] == 1
        assert len(mgr.history) == 1
        assert mgr.can_undo

    def test_undo(self) -> None:
        mgr = UndoRedoManager()
        counter = [0]
        mgr.execute(IncrementCommand(counter))
        mgr.undo()
        assert counter[0] == 0
        assert not mgr.can_undo
        assert mgr.can_redo

    def test_redo(self) -> None:
        mgr = UndoRedoManager()
        counter = [0]
        mgr.execute(IncrementCommand(counter))
        mgr.undo()
        mgr.redo()
        assert counter[0] == 1
        assert mgr.can_undo
        assert not mgr.can_redo

    def test_undo_nothing_returns_none(self) -> None:
        mgr = UndoRedoManager()
        assert mgr.undo() is None

    def test_redo_nothing_returns_none(self) -> None:
        mgr = UndoRedoManager()
        assert mgr.redo() is None

    def test_execute_clears_redo_stack(self) -> None:
        mgr = UndoRedoManager()
        counter = [0]
        mgr.execute(IncrementCommand(counter))
        mgr.execute(IncrementCommand(counter))
        mgr.undo()
        assert mgr.can_redo
        # New execute should clear redo
        mgr.execute(IncrementCommand(counter, amount=10))
        assert not mgr.can_redo

    def test_max_history(self) -> None:
        mgr = UndoRedoManager(max_history=3)
        counter = [0]
        for _ in range(5):
            mgr.execute(IncrementCommand(counter))
        # At most 3 in history
        assert len(mgr.history) <= 3

    def test_clear(self) -> None:
        mgr = UndoRedoManager()
        counter = [0]
        mgr.execute(IncrementCommand(counter))
        mgr.clear()
        assert not mgr.can_undo
        assert not mgr.can_redo
        assert mgr.history == []

    def test_undo_description(self) -> None:
        mgr = UndoRedoManager()
        assert mgr.undo_description is None
        mgr.execute(IncrementCommand([0], 7))
        desc = mgr.undo_description
        assert desc is not None

    def test_redo_description(self) -> None:
        mgr = UndoRedoManager()
        mgr.execute(IncrementCommand([0]))
        mgr.undo()
        desc = mgr.redo_description
        assert desc is not None

    def test_listener_notification(self) -> None:
        mgr = UndoRedoManager()
        events: list[str] = []
        mgr.add_listener(lambda e: events.append(e))
        mgr.execute(IncrementCommand([0]))
        assert len(events) > 0

    def test_remove_listener(self) -> None:
        mgr = UndoRedoManager()
        events: list[str] = []
        cb = lambda e: events.append(e)  # noqa: E731
        mgr.add_listener(cb)
        mgr.remove_listener(cb)
        mgr.execute(IncrementCommand([0]))
        assert len(events) == 0

    def test_multiple_undo_redo_cycles(self) -> None:
        mgr = UndoRedoManager()
        counter = [0]
        mgr.execute(IncrementCommand(counter, 1))
        mgr.execute(IncrementCommand(counter, 2))
        mgr.execute(IncrementCommand(counter, 3))
        assert counter[0] == 6
        mgr.undo()
        assert counter[0] == 3
        mgr.undo()
        assert counter[0] == 1
        mgr.redo()
        assert counter[0] == 3
