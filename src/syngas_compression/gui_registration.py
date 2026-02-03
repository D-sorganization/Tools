"""GUI Registration for Syngas Compression Calculator.

Registers the calculator with the shared GUI launcher system.
"""

from __future__ import annotations

from pathlib import Path

try:
    from shared.python.gui_launcher import GUIType, LaunchConfig, register_gui

    # Register Syngas Compression Calculator
    MODULE_DIR = Path(__file__).parent

    register_gui(
        tool_name="syngas_compression",
        display_name="Syngas Compression Calculator",
        description="Multi-stage compression analysis with water dropout calculations",
        gui_configs={
            GUIType.PYQT6: LaunchConfig(
                tool_name="syngas_compression",
                gui_type=GUIType.PYQT6,
                script_path=str(MODULE_DIR / "launch_pyqt6.py"),
                title="Syngas Compression Calculator",
                dependencies=["PyQt6", "matplotlib"],
            ),
            GUIType.REACT: LaunchConfig(
                tool_name="syngas_compression",
                gui_type=GUIType.REACT,
                script_path=str(MODULE_DIR / "launch_web.py"),
                title="Syngas Compression Calculator (Web)",
                working_directory=str(MODULE_DIR / "web"),
                dev_command="npm run dev",
                port=5173,
            ),
        },
        category="Process Simulation",
    )
except ImportError:
    # GUI launcher not available, skip registration
    pass
