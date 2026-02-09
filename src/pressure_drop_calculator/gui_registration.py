"""GUI Registration for Pressure Drop Calculator."""

from __future__ import annotations

from pathlib import Path

try:
    from gui_launcher import GUIType, LaunchConfig, register_gui

    MODULE_DIR = Path(__file__).parent

    register_gui(
        tool_name="pressure_drop_calculator",
        display_name="Pressure Drop Calculator",
        description="Pipe flow pressure drop analysis with multiple friction methods",
        gui_configs={
            GUIType.PYQT6: LaunchConfig(
                tool_name="pressure_drop_calculator",
                gui_type=GUIType.PYQT6,
                script_path=str(MODULE_DIR / "launch_pyqt6.py"),
                title="Pressure Drop Calculator",
                dependencies=["PyQt6", "matplotlib"],
            ),
            GUIType.REACT: LaunchConfig(
                tool_name="pressure_drop_calculator",
                gui_type=GUIType.REACT,
                script_path=str(MODULE_DIR / "launch_web.py"),
                title="Pressure Drop Calculator (Web)",
                working_directory=str(MODULE_DIR / "web"),
                dev_command="npm run dev",
                port=5175,
            ),
        },
        category="Process Simulation",
    )
except ImportError:
    pass
