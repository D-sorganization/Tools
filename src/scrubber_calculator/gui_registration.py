"""GUI Registration for Scrubber Calculator."""

from __future__ import annotations

from pathlib import Path

try:
    from gui_launcher import GUIType, LaunchConfig, register_gui

    MODULE_DIR = Path(__file__).parent

    register_gui(
        tool_name="scrubber_calculator",
        display_name="Scrubber Calculator",
        description="Packed bed scrubber design with NTU/HTU mass transfer",
        gui_configs={
            GUIType.PYQT6: LaunchConfig(
                tool_name="scrubber_calculator",
                gui_type=GUIType.PYQT6,
                script_path=str(MODULE_DIR / "launch_pyqt6.py"),
                title="Scrubber Calculator",
                dependencies=["PyQt6", "matplotlib"],
            ),
            GUIType.REACT: LaunchConfig(
                tool_name="scrubber_calculator",
                gui_type=GUIType.REACT,
                script_path=str(MODULE_DIR / "launch_web.py"),
                title="Scrubber Calculator (Web)",
                working_directory=str(MODULE_DIR / "web"),
                dev_command="npm run dev",
                port=5177,
            ),
        },
        category="Process Simulation",
    )
except ImportError:
    pass
