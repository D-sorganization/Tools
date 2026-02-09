"""GUI Registration for WGS Reactor Calculator."""

from __future__ import annotations

from pathlib import Path

try:
    from gui_launcher import GUIType, LaunchConfig, register_gui

    MODULE_DIR = Path(__file__).parent

    register_gui(
        tool_name="wgs_reactor",
        display_name="WGS Reactor Calculator",
        description="Water-Gas Shift reactor equilibrium and sizing calculations",
        gui_configs={
            GUIType.PYQT6: LaunchConfig(
                tool_name="wgs_reactor",
                gui_type=GUIType.PYQT6,
                script_path=str(MODULE_DIR / "launch_pyqt6.py"),
                title="WGS Reactor Calculator",
                dependencies=["PyQt6"],
            ),
            GUIType.REACT: LaunchConfig(
                tool_name="wgs_reactor",
                gui_type=GUIType.REACT,
                script_path=str(MODULE_DIR / "launch_web.py"),
                title="WGS Reactor Calculator (Web)",
                working_directory=str(MODULE_DIR / "web"),
                dev_command="npm run dev",
                port=5178,
            ),
        },
        category="Process Simulation",
    )
except ImportError:
    pass
