"""Pressure Drop Calculator Module.

Provides standalone GUI interfaces for pipe pressure drop analysis.
Uses the shared engine from ``sidekick.process_calculators``.

Import contract:
    The pressure-drop engine lives in ``sidekick`` (``src/shared/python``),
    which is not guaranteed to be on ``sys.path`` for cross-repo consumers
    that only place the repository root on the path. To keep
    ``import src.pressure_drop_calculator`` working in that configuration,
    the heavy ``sidekick`` symbols are resolved lazily via :pep:`562`
    ``__getattr__`` on first attribute access rather than at package import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shared.python.sidekick.process_calculators.pressure_drop_calculator import (
        PressureDropCalculationEngine,
        PressureDropInputs,
        PressureDropResults,
        calculate_pressure_drop,
    )

_EXPORTS = (
    "PressureDropCalculationEngine",
    "PressureDropInputs",
    "PressureDropResults",
    "calculate_pressure_drop",
)


def __getattr__(name: str) -> Any:
    """Lazily import pressure-drop symbols from ``sidekick`` (:pep:`562`)."""
    if name in _EXPORTS:
        import importlib

        module = importlib.import_module(
            "sidekick.process_calculators.pressure_drop_calculator"
        )
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily-exposed attributes in ``dir()``."""
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "PressureDropCalculationEngine",
    "PressureDropInputs",
    "PressureDropResults",
    "calculate_pressure_drop",
]
