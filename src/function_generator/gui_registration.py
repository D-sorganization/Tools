"""GUI Registration for Function Generator.

Registers the function generator with the shared GUI launcher system.
"""

from __future__ import annotations

from pathlib import Path

try:
    from shared.python.gui_launcher import GUIType, LaunchConfig, register_gui

    MODULE_DIR = Path(__file__).parent

    register_gui(
        tool_name="function_generator",
        display_name="Function Generator",
        description="Generate and visualize various waveforms (sine, square, triangle, etc.)",
        gui_configs={
            GUIType.PYQT6: LaunchConfig(
                tool_name="function_generator",
                gui_type=GUIType.PYQT6,
                script_path=str(MODULE_DIR / "launch_pyqt6.py"),
                title="Function Generator",
                dependencies=["PyQt6", "matplotlib", "numpy"],
            ),
            GUIType.REACT: LaunchConfig(
                tool_name="function_generator",
                gui_type=GUIType.REACT,
                script_path=str(MODULE_DIR / "launch_web.py"),
                title="Function Generator (Web)",
                working_directory=str(MODULE_DIR / "web"),
                dev_command="npm run dev",
                port=5174,
            ),
        },
        category="Signal Processing",
    )
except ImportError:
    pass
