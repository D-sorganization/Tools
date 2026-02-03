"""UI components for Data Processor.

This module provides both PyQt6 and React-compatible UI components.
"""

from .pyqt6 import DataProcessorMainWindow, FilterConfigWidget, SignalListWidget

__all__ = [
    "DataProcessorMainWindow",
    "FilterConfigWidget",
    "SignalListWidget",
]
