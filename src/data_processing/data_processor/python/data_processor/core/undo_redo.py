"""Undo/Redo system using the Command pattern.

Provides a flexible undo/redo mechanism for data processing operations.
Each operation is encapsulated as a Command object that knows how to
execute and reverse itself.

Design Patterns:
- Command Pattern: Encapsulates operations as objects
- Memento Pattern: State snapshots for complex operations
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Command(ABC):
    """Abstract base class for undoable commands.

    Each command encapsulates an operation that can be executed
    and reversed. Commands are immutable after creation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for the command."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description of what the command does."""

    @abstractmethod
    def execute(self) -> Any:
        """Execute the command and return the result."""

    @abstractmethod
    def undo(self) -> Any:
        """Reverse the command and return the previous state."""

    def redo(self) -> Any:
        """Re-execute the command. Default implementation calls execute()."""
        return self.execute()


@dataclass
class CommandRecord:
    """Record of an executed command with timestamp."""

    command: Command
    executed_at: datetime = field(default_factory=datetime.now)
    result: Any = None


class UndoRedoManager(Generic[T]):
    """Manages undo/redo operations for a sequence of commands.

    Type parameter T represents the state type being managed.
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize the undo/redo manager.

        Args:
            max_history: Maximum number of commands to keep in history
        """
        if not (max_history is not None):
            raise ValueError("max_history must be provided")
        self._history: list[CommandRecord] = []
        self._redo_stack: list[CommandRecord] = []
        self._max_history = max_history
        self._listeners: list[Callable[[str], None]] = []

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._history) > 0

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    @property
    def undo_description(self) -> str | None:
        """Get description of the command that would be undone."""
        if self._history:
            return self._history[-1].command.name
        return None

    @property
    def redo_description(self) -> str | None:
        """Get description of the command that would be redone."""
        if self._redo_stack:
            return self._redo_stack[-1].command.name
        return None

    @property
    def history(self) -> list[str]:
        """Get list of command names in history."""
        return [record.command.name for record in self._history]

    def execute(self, command: Command) -> Any:
        """Execute a command and add it to history.

        Args:
            command: The command to execute

        Returns:
            Result of the command execution
        """
        if not (command is not None):
            raise ValueError("command must be provided")
        result = command.execute()

        record = CommandRecord(command=command, result=result)
        self._history.append(record)

        # Clear redo stack when new command is executed
        self._redo_stack.clear()

        # Trim history if exceeds max
        if len(self._history) > self._max_history:
            self._history.pop(0)

        self._notify_listeners("execute")
        logger.debug(f"Executed: {command.name}")
        return result

    def undo(self) -> Any | None:
        """Undo the last command.

        Returns:
            Result of the undo operation, or None if nothing to undo
        """
        if not self.can_undo:
            return None

        record = self._history.pop()
        result = record.command.undo()
        self._redo_stack.append(record)

        self._notify_listeners("undo")
        logger.debug(f"Undone: {record.command.name}")
        return result

    def redo(self) -> Any | None:
        """Redo the last undone command.

        Returns:
            Result of the redo operation, or None if nothing to redo
        """
        if not self.can_redo:
            return None

        record = self._redo_stack.pop()
        result = record.command.redo()
        self._history.append(record)

        self._notify_listeners("redo")
        logger.debug(f"Redone: {record.command.name}")
        return result

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()
        self._redo_stack.clear()
        self._notify_listeners("clear")

    def add_listener(self, listener: Callable[[str], None]) -> None:
        """Add a listener for undo/redo events.

        Args:
            listener: Callback function that receives event type
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[str], None]) -> None:
        """Remove a listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self, event_type: str) -> None:
        """Notify all listeners of an event."""
        for listener in self._listeners:
            try:
                listener(event_type)
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning(f"Listener error: {e}")


# =============================================================================
# DATAFRAME-SPECIFIC COMMANDS
# =============================================================================


class DataFrameCommand(Command):
    """Base class for DataFrame operations."""

    def __init__(
        self,
        df_getter: Callable[[], pd.DataFrame],
        df_setter: Callable[[pd.DataFrame], None],
    ) -> None:
        """Initialize with DataFrame accessor functions.

        Args:
            df_getter: Function to get current DataFrame
            df_setter: Function to set new DataFrame
        """
        if not (df_getter is not None):
            raise ValueError("df_getter must be provided")
        self._get_df = df_getter
        self._set_df = df_setter
        self._previous_df: pd.DataFrame | None = None


class FilterCommand(DataFrameCommand):
    """Command for applying filters to data."""

    def __init__(
        self,
        df_getter: Callable[[], pd.DataFrame],
        df_setter: Callable[[pd.DataFrame], None],
        filter_func: Callable[[pd.DataFrame], pd.DataFrame],
        filter_name: str,
        filter_params: dict[str, Any],
    ) -> None:
        if not (df_getter is not None):
            raise ValueError("df_getter must be provided")
        super().__init__(df_getter, df_setter)
        self._filter_func = filter_func
        self._filter_name = filter_name
        self._filter_params = filter_params

    @property
    def name(self) -> str:
        return f"Apply {self._filter_name}"

    @property
    def description(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self._filter_params.items())
        return f"Apply {self._filter_name} filter with parameters: {params_str}"

    def execute(self) -> pd.DataFrame:
        self._previous_df = self._get_df().copy()
        result = self._filter_func(self._previous_df)
        self._set_df(result)
        return result

    def undo(self) -> pd.DataFrame:
        if self._previous_df is not None:
            self._set_df(self._previous_df)
            return self._previous_df
        raise RuntimeError("Cannot undo: no previous state saved")


class ColumnOperationCommand(DataFrameCommand):
    """Command for adding/removing/modifying columns."""

    def __init__(
        self,
        df_getter: Callable[[], pd.DataFrame],
        df_setter: Callable[[pd.DataFrame], None],
        operation: str,
        column_name: str,
        new_data: pd.Series | None = None,
        formula: str | None = None,
    ) -> None:
        if not (df_getter is not None):
            raise ValueError("df_getter must be provided")
        super().__init__(df_getter, df_setter)
        self._operation = operation  # "add", "remove", "rename", "modify"
        self._column_name = column_name
        self._new_data = new_data
        self._formula = formula
        self._old_data: pd.Series | None = None

    @property
    def name(self) -> str:
        return f"{self._operation.capitalize()} column '{self._column_name}'"

    @property
    def description(self) -> str:
        if self._formula:
            return (
                f"{self._operation} column '{self._column_name}'" f" using formula: {self._formula}"
            )
        return f"{self._operation} column '{self._column_name}'"

    def execute(self) -> pd.DataFrame:
        df = self._get_df().copy()

        if self._operation == "add":
            if self._new_data is not None:
                df[self._column_name] = self._new_data
        elif self._operation == "remove":
            if self._column_name in df.columns:
                self._old_data = df[self._column_name].copy()
                df = df.drop(columns=[self._column_name])
        elif self._operation == "modify":
            if self._column_name in df.columns:
                self._old_data = df[self._column_name].copy()
            if self._new_data is not None:
                df[self._column_name] = self._new_data

        self._set_df(df)
        return df

    def undo(self) -> pd.DataFrame:
        df = self._get_df().copy()

        if self._operation == "add":
            if self._column_name in df.columns:
                df = df.drop(columns=[self._column_name])
        elif self._operation == "remove":
            if self._old_data is not None:
                df[self._column_name] = self._old_data
        elif self._operation == "modify":
            if self._old_data is not None:
                df[self._column_name] = self._old_data

        self._set_df(df)
        return df


class RowFilterCommand(DataFrameCommand):
    """Command for filtering rows based on conditions."""

    def __init__(
        self,
        df_getter: Callable[[], pd.DataFrame],
        df_setter: Callable[[pd.DataFrame], None],
        mask: pd.Series,
        filter_description: str,
    ) -> None:
        if not (df_getter is not None):
            raise ValueError("df_getter must be provided")
        super().__init__(df_getter, df_setter)
        self._mask = mask
        self._filter_description = filter_description

    @property
    def name(self) -> str:
        return f"Filter rows: {self._filter_description}"

    @property
    def description(self) -> str:
        return f"Filter rows where {self._filter_description}"

    def execute(self) -> pd.DataFrame:
        self._previous_df = self._get_df().copy()
        result = self._previous_df[self._mask].reset_index(drop=True)
        self._set_df(result)
        return result

    def undo(self) -> pd.DataFrame:
        if self._previous_df is not None:
            self._set_df(self._previous_df)
            return self._previous_df
        raise RuntimeError("Cannot undo: no previous state saved")


class ResampleCommand(DataFrameCommand):
    """Command for resampling time series data."""

    def __init__(
        self,
        df_getter: Callable[[], pd.DataFrame],
        df_setter: Callable[[pd.DataFrame], None],
        resample_func: Callable[[pd.DataFrame], pd.DataFrame],
        rule: str,
        method: str,
    ) -> None:
        if not (df_getter is not None):
            raise ValueError("df_getter must be provided")
        super().__init__(df_getter, df_setter)
        self._resample_func = resample_func
        self._rule = rule
        self._method = method

    @property
    def name(self) -> str:
        return f"Resample to {self._rule}"

    @property
    def description(self) -> str:
        return f"Resample data to {self._rule} using {self._method} aggregation"

    def execute(self) -> pd.DataFrame:
        self._previous_df = self._get_df().copy()
        result = self._resample_func(self._previous_df)
        self._set_df(result)
        return result

    def undo(self) -> pd.DataFrame:
        if self._previous_df is not None:
            self._set_df(self._previous_df)
            return self._previous_df
        raise RuntimeError("Cannot undo: no previous state saved")


class CompositeCommand(Command):
    """Command that executes multiple commands as a single unit.

    Useful for grouping related operations that should be
    undone/redone together.
    """

    def __init__(self, commands: list[Command], group_name: str) -> None:
        """Initialize with a list of commands.

        Args:
            commands: List of commands to execute as a group
            group_name: Name for the composite operation
        """
        if not (commands is not None):
            raise ValueError("commands must be provided")
        self._commands = commands
        self._group_name = group_name

    @property
    def name(self) -> str:
        return self._group_name

    @property
    def description(self) -> str:
        sub_descriptions = [cmd.name for cmd in self._commands]
        return f"{self._group_name}: {', '.join(sub_descriptions)}"

    def execute(self) -> list[Any]:
        """Execute all commands in order."""
        results = []
        for cmd in self._commands:
            results.append(cmd.execute())
        return results

    def undo(self) -> list[Any]:
        """Undo all commands in reverse order."""
        results = []
        for cmd in reversed(self._commands):
            results.append(cmd.undo())
        return results


class LambdaCommand(Command):
    """Simple command created from lambda functions.

    Useful for quick one-off operations.
    """

    def __init__(
        self,
        execute_fn: Callable[[], Any],
        undo_fn: Callable[[], Any],
        name: str,
        description: str = "",
    ) -> None:
        if not (execute_fn is not None):
            raise ValueError("execute_fn must be provided")
        self._execute_fn = execute_fn
        self._undo_fn = undo_fn
        self._name = name
        self._description = description or name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self) -> Any:
        return self._execute_fn()

    def undo(self) -> Any:
        return self._undo_fn()


__all__ = [
    "Command",
    "CommandRecord",
    "UndoRedoManager",
    "DataFrameCommand",
    "FilterCommand",
    "ColumnOperationCommand",
    "RowFilterCommand",
    "ResampleCommand",
    "CompositeCommand",
    "LambdaCommand",
]
