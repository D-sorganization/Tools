"""Shared GUI Launcher Infrastructure.

This module provides a unified launcher infrastructure that supports:
- PyQt6 desktop applications
- React/Vite web applications
- Cross-repository component reuse

The launcher handles dependency checking, environment setup, and provides
consistent launch mechanisms across different GUI frameworks.

Example:
    from gui_launcher import GUILauncher, GUIType

    # Launch PyQt6 app
    launcher = GUILauncher(tool_name="data_processor", gui_type=GUIType.PYQT6)
    launcher.launch()

    # Launch React web app
    launcher = GUILauncher(tool_name="data_processor", gui_type=GUIType.REACT)
    launcher.launch()
"""

from .launcher import GUILauncher, GUIType, LaunchConfig
from .registry import GUIRegistry, get_registry, register_gui

__all__ = [
    "GUILauncher",
    "GUIType",
    "LaunchConfig",
    "GUIRegistry",
    "register_gui",
    "get_registry",
]
