"""GUI Registration for Data Processor.

This module registers the Data Processor GUI variants with the shared
GUI launcher infrastructure, enabling discovery and launch from
tile launchers and other tools.
"""

from __future__ import annotations

from pathlib import Path

# Try to import from shared gui_launcher
try:
    from gui_launcher import (
        GUIType,
        LaunchConfig,
        register_gui,
    )

    # Get paths relative to this file
    BASE_PATH = Path(__file__).parent

    # PyQt6 Configuration
    pyqt6_config = LaunchConfig(
        tool_name="data_processor",
        gui_type=GUIType.PYQT6,
        module_path="data_processor.ui.pyqt6.main_window",
        entry_point=str(BASE_PATH / "launch_pyqt6.py"),
        dependencies=["PyQt6", "pandas", "numpy"],
        working_dir=str(BASE_PATH / "python"),
    )

    # React Web Configuration
    react_config = LaunchConfig(
        tool_name="data_processor",
        gui_type=GUIType.REACT,
        web_path=str(BASE_PATH / "web"),
        entry_point=str(BASE_PATH / "launch_web.py"),
        port=3000,
        auto_open_browser=True,
    )

    # Register both GUIs
    register_gui(
        tool_name="data_processor",
        display_name="Data Processor",
        description="Signal processing and time-series data analysis tool",
        gui_configs={
            GUIType.PYQT6: pyqt6_config,
            GUIType.REACT: react_config,
        },
        category="Data Processing",
        repository="Tools",
    )

except ImportError:
    # Shared launcher not available, skip registration
    pass


# Standalone registration function for manual use
def get_gui_configs() -> dict:
    """Get GUI configurations for manual launcher integration.

    Returns:
        Dictionary with GUI type keys and configuration dicts
    """
    base_path = Path(__file__).parent

    return {
        "pyqt6": {
            "name": "Data Processor (PyQt6)",
            "path": str(base_path / "launch_pyqt6.py"),
            "type": "python",
            "description": "Desktop signal processing application",
            "dependencies": ["PyQt6", "pandas", "numpy"],
        },
        "react": {
            "name": "Data Processor (Web)",
            "path": str(base_path / "launch_web.py"),
            "type": "python",
            "description": "Web-based signal processing application",
            "dependencies": ["node", "npm"],
        },
    }


if __name__ == "__main__":
    # Print registration info when run directly
    import json

    print("Data Processor GUI Registration")
    print("=" * 40)
    print(json.dumps(get_gui_configs(), indent=2))
