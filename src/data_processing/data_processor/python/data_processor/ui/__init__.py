"""UI components for Data Processor."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DataProcessorMainWindow",
    "FilterConfigWidget",
    "SignalListWidget",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .pyqt6 import DataProcessorMainWindow, FilterConfigWidget, SignalListWidget

        exports = {
            "DataProcessorMainWindow": DataProcessorMainWindow,
            "FilterConfigWidget": FilterConfigWidget,
            "SignalListWidget": SignalListWidget,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
