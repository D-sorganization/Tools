"""GUI Registration for Electrode Advisor.

Registers Electrode Advisor GUI configurations with the shared launcher system.
"""

from __future__ import annotations

from pathlib import Path

from gui_launcher import GUIType, LaunchConfig, register_gui

# Get the current directory for relative paths
CURRENT_DIR = Path(__file__).parent


def register_electrode_advisor() -> None:
    """Register Electrode Advisor GUIs with the launcher system."""
    register_gui(
        tool_name="electrode_advisor",
        display_name="Electrode Advisor",
        description="AC Electrode Advancement Module for electrode system analysis",
        gui_configs={
            GUIType.PYQT6: LaunchConfig(
                tool_name="electrode_advisor",
                gui_type=GUIType.PYQT6,
                module_path="electrode_advisor.ui.pyqt6.main_window",
                entry_point=str(CURRENT_DIR / "launch_pyqt6.py"),
                dependencies=["PyQt6", "numpy", "matplotlib"],
            ),
            GUIType.REACT: LaunchConfig(
                tool_name="electrode_advisor",
                gui_type=GUIType.REACT,
                web_path=str(CURRENT_DIR / "web"),
                port=3001,
            ),
        },
        category="Process Simulation",
        repository="Tools",
    )


# Auto-register when module is imported
register_electrode_advisor()
