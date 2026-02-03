"""GUI Registration for TRC Vessel Designer."""

from __future__ import annotations

from pathlib import Path

from shared.python.gui_launcher import GUIType, LaunchConfig, register_gui

CURRENT_DIR = Path(__file__).parent


def register_trc_vessel_designer() -> None:
    """Register TRC Vessel Designer GUIs with the launcher system."""
    register_gui(
        tool_name="trc_vessel_designer",
        display_name="TRC Vessel Designer",
        description="Thermal Reaction Chamber vessel design tool",
        gui_configs={
            GUIType.PYQT6: LaunchConfig(
                tool_name="trc_vessel_designer",
                gui_type=GUIType.PYQT6,
                module_path="trc_vessel_designer.ui.pyqt6.main_window",
                entry_point=str(CURRENT_DIR / "launch_pyqt6.py"),
                dependencies=["PyQt6", "numpy"],
            ),
            GUIType.REACT: LaunchConfig(
                tool_name="trc_vessel_designer",
                gui_type=GUIType.REACT,
                web_path=str(CURRENT_DIR / "web"),
                port=3002,
            ),
        },
        category="Process Simulation",
        repository="Tools",
    )


register_trc_vessel_designer()
