"""GUI Registration for Acid Gas Dewpoint Calculator."""

from __future__ import annotations

from pathlib import Path

try:
    from shared.python.gui_launcher import GUIType, LaunchConfig, register_gui

    MODULE_DIR = Path(__file__).parent

    register_gui(
        tool_name="acid_gas_dewpoint",
        display_name="Acid Gas Dewpoint Calculator",
        description="HF, HCl, H2S dewpoint analysis for syngas applications",
        gui_configs={
            GUIType.PYQT6: LaunchConfig(
                tool_name="acid_gas_dewpoint",
                gui_type=GUIType.PYQT6,
                script_path=str(MODULE_DIR / "launch_pyqt6.py"),
                title="Acid Gas Dewpoint Calculator",
                dependencies=["PyQt6", "matplotlib", "numpy"],
            ),
            GUIType.REACT: LaunchConfig(
                tool_name="acid_gas_dewpoint",
                gui_type=GUIType.REACT,
                script_path=str(MODULE_DIR / "launch_web.py"),
                title="Acid Gas Dewpoint Calculator (Web)",
                working_directory=str(MODULE_DIR / "web"),
                dev_command="npm run dev",
                port=5176,
            ),
        },
        category="Process Simulation",
    )
except ImportError:
    pass
