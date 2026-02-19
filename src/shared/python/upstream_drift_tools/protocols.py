"""Protocol interfaces for upstream drift tools.

These protocols define the structural typing contracts for process
calculators, data processors, and UI state management. They enable
loose coupling between the shared calculation engines, GUI layers,
and the FastAPI backend.

Usage:
    def size_equipment(calc: ProcessCalculator) -> dict[str, float]:
        inputs = calc.get_default_inputs()
        return calc.calculate(inputs)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProcessCalculator(Protocol):
    """Protocol for process engineering calculators.

    All calculators in ``process_calculators/`` share the pattern of
    accepting a dict (or dataclass) of inputs and returning a dict
    (or dataclass) of results.

    Implementations: FlareCalculator, BaghouseCalculator,
    AcidGasDewpointCalculator, FinancialCalculator, etc.
    """

    def calculate(self, inputs: Any) -> Any:
        """Run the calculation and return results."""
        ...


@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for tabular data transformation operations.

    Covers the common pattern in DataProcessor and its pipeline
    stages: accept data, transform it, return the result.
    """

    def transform(self, data: Any) -> Any:
        """Apply a transformation and return the result."""
        ...


@runtime_checkable
class StateSerializable(Protocol):
    """Protocol for objects that can save/restore UI state.

    Used by CalculatorStateMixin and any widget that persists
    its configuration across sessions.
    """

    def save_state(self) -> dict[str, Any]:
        """Serialize current state to a dictionary."""
        ...

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore state from a previously serialized dictionary."""
        ...


@runtime_checkable
class UnitConverter(Protocol):
    """Protocol for unit conversion operations.

    Implementations should convert a numeric value from one unit
    to another within the same physical dimension.
    """

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert *value* from *from_unit* to *to_unit*."""
        ...
