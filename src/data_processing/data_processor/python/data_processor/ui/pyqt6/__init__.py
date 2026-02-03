"""PyQt6 GUI components for Data Processor."""

from .main_window import DataProcessorMainWindow
from .widgets import FilterConfigWidget, SignalListWidget, StatisticsWidget

__all__ = [
    "DataProcessorMainWindow",
    "FilterConfigWidget",
    "SignalListWidget",
    "StatisticsWidget",
]
