"""PyQt6-based GUI for the Data Processor.

This package provides a clean architecture GUI with:
- Widgets: Small, focused UI components
- Presenters: Connect widgets to business logic
- Styles: Consistent theming
"""

from .main_window import DataProcessorMainWindow

__all__ = ["DataProcessorMainWindow"]
