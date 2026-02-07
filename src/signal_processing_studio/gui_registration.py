"""GUI Registration for Signal Processing Studio.

Registers the studio with the shared GUI launcher system.
"""

from __future__ import annotations

from pathlib import Path

try:
    from shared.python.gui_launcher import GUIType, LaunchConfig, register_gui

    MODULE_DIR = Path(__file__).parent

    register_gui(
        tool_name="signal_processing_studio",
        display_name="Signal Processing Studio",
        description=(
            "Unified signal processing: waveform generation, "
            "analysis, filtering, curve fitting, polynomial design"
        ),
        gui_configs={
            GUIType.PYQT6: LaunchConfig(
                tool_name="signal_processing_studio",
                gui_type=GUIType.PYQT6,
                script_path=str(MODULE_DIR / "launch_pyqt6.py"),
                title="Signal Processing Studio",
                dependencies=[
                    "PyQt6",
                    "matplotlib",
                    "numpy",
                    "scipy",
                    "sympy",
                ],
            ),
        },
        category="Signal Processing",
    )
except ImportError:
    pass
