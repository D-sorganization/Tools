"""GUI registration for Flare Calculator."""

from gui_launcher import GUIType, LaunchConfig, register_gui


def register_flare_calculator_guis() -> None:
    """Register Flare Calculator GUI interfaces."""
    register_gui(
        LaunchConfig(
            name="Flare Calculator (Desktop)",
            module_path="flare_calculator.ui.pyqt6.main_window",
            class_name="FlareCalculatorMainWindow",
            gui_type=GUIType.PYQT6,
            description="Flare sizing and safety zone calculator",
            category="Process Simulation",
        )
    )

    register_gui(
        LaunchConfig(
            name="Flare Calculator (Web)",
            module_path="flare_calculator",
            gui_type=GUIType.REACT,
            description="Flare sizing and safety zone calculator (web)",
            category="Process Simulation",
            web_port=5179,
        )
    )


# Auto-register on import
register_flare_calculator_guis()
