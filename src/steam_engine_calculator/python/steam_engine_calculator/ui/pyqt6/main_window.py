"""
Steam Engine Calculator - PyQt6 Main Window
============================================

Full-featured GUI for steam thermodynamic property calculations.
Uses Catppuccin Mocha dark theme for modern appearance.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from upstream_drift_tools.ui.widgets.base_calculator_widget import BaseCalculatorWindow

if TYPE_CHECKING:
    from upstream_drift_tools.calculators.thermo.steam_engine import SteamProperties

logger = logging.getLogger(__name__)

# LoD: module-level aliases for Qt enum chains (avoids 3-level deep attribute access)
_SCROLL_BAR_AS_NEEDED = Qt.ScrollBarPolicy.ScrollBarAsNeeded
_FONT_WEIGHT_BOLD = QFont.Weight.Bold
_ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
_FRAME_STYLED_PANEL = QFrame.Shape.StyledPanel

# Catppuccin Mocha color palette
COLORS = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "subtext1": "#bac2de",
    "blue": "#89b4fa",
    "green": "#a6e3a1",
    "red": "#f38ba8",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "mauve": "#cba6f7",
    "teal": "#94e2d5",
    "lavender": "#b4befe",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
}


def validate_temperature_k(value: float) -> tuple[bool, str]:
    """Validate temperature in Kelvin."""
    if value < 273.16:
        return False, "Temperature below triple point (273.16 K)"
    if value > 647.15:
        return False, "Temperature above critical point (647.15 K)"
    return True, ""


def validate_pressure_pa(value: float) -> tuple[bool, str]:
    """Validate pressure in Pascals."""
    if value <= 0:
        return False, "Pressure must be positive"
    if value > 100e6:
        return False, "Pressure exceeds maximum (100 MPa)"
    return True, ""


def format_temperature(value: float, unit: str = "K") -> str:
    """Format temperature with units.

    Args:
        value: Temperature value (numeric).
        unit: Unit string, "K" or "C".

    Raises:
        TypeError: If value is not a number.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"value must be a number, got {type(value).__name__}")
    if unit == "C":
        return f"{value - 273.15:.2f} °C"
    return f"{value:.2f} K"


def format_pressure(value: float, unit: str = "Pa") -> str:
    """Format pressure with units.

    Args:
        value: Pressure value (numeric).
        unit: Unit string, "Pa", "kPa", "bar", or "MPa".

    Raises:
        TypeError: If value is not a number.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"value must be a number, got {type(value).__name__}")
    if unit == "bar":
        return f"{value / 1e5:.4f} bar"
    if unit == "kPa":
        return f"{value / 1000:.2f} kPa"
    if unit == "MPa":
        return f"{value / 1e6:.4f} MPa"
    return f"{value:.2f} Pa"


def format_enthalpy(value: float) -> str:
    """Format enthalpy (J/kg to kJ/kg)."""
    return f"{value / 1000:.2f} kJ/kg"


def format_entropy(value: float) -> str:
    """Format entropy (J/kg-K to kJ/kg-K)."""
    return f"{value / 1000:.4f} kJ/kg-K"


class SteamEngineCalculatorWindow(BaseCalculatorWindow):
    """Main window for the Steam Engine Calculator."""

    # Calculation mode constants
    MODE_TP = 0  # Temperature and Pressure
    MODE_SAT_T = 1  # Saturated from Temperature
    MODE_SAT_P = 2  # Saturated from Pressure

    # Input field definitions
    INPUT_FIELDS = ["temperature", "pressure"]

    # Result field definitions
    RESULT_FIELDS = [
        "temperature",
        "pressure",
        "density",
        "specific_volume",
        "enthalpy",
        "entropy",
        "internal_energy",
        "cp",
        "cv",
        "speed_of_sound",
        "thermal_conductivity",
        "dynamic_viscosity",
        "kinematic_viscosity",
        "quality",
        "phase",
        "compressibility_factor",
        "prandtl_number",
        "specific_heat_ratio",
    ]

    def __init__(self) -> None:
        super().__init__(
            calculator_name="SteamEngineCalculator",
            window_title="Steam Engine Calculator",
            min_size=(900, 700),
        )
        self.engine: Any = None
        self.result_labels: dict[str, QLabel] = {}
        self._init_engine()
        self._init_ui()
        self._apply_styles()

    def _init_engine(self) -> None:
        """Initialize the calculation engine."""
        try:
            from upstream_drift_tools.calculators.thermo.steam_engine import (
                SteamCalculationEngine,
            )

            self.engine = SteamCalculationEngine()
        except ImportError:
            self.engine = None

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        # Scroll area wrapping the content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(_SCROLL_BAR_AS_NEEDED)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Steam Thermodynamic Properties Calculator")
        title.setFont(QFont("Segoe UI", 18, _FONT_WEIGHT_BOLD))
        title.setStyleSheet(f"color: {COLORS['blue']};")
        main_layout.addWidget(title)

        # Content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left panel - Inputs
        left_panel = self._create_input_panel()
        content_layout.addWidget(left_panel, 1)

        # Right panel - Results
        right_panel = self._create_results_panel()
        content_layout.addWidget(right_panel, 2)

        main_layout.addLayout(content_layout)
        scroll.setWidget(central)
        self.main_layout.addWidget(scroll)

    def _create_input_panel(self) -> QWidget:
        """Create the input panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        # Calculation Mode
        mode_group = self._create_group("Calculation Mode")
        mode_layout = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                "Temperature & Pressure",
                "Saturated (from Temperature)",
                "Saturated (from Pressure)",
            ]
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Input Parameters
        input_group = self._create_group("Input Parameters")
        input_layout = QGridLayout()
        input_layout.setSpacing(10)

        # Temperature input
        input_layout.addWidget(self._create_label("Temperature:"), 0, 0)
        self.temp_input = QLineEdit("373.15")
        self.temp_input.setPlaceholderText("K")
        input_layout.addWidget(self.temp_input, 0, 1)

        self.temp_unit = QComboBox()
        self.temp_unit.addItems(["K", "°C"])
        self.temp_unit.currentIndexChanged.connect(self._convert_temp_display)
        input_layout.addWidget(self.temp_unit, 0, 2)

        # Pressure input
        input_layout.addWidget(self._create_label("Pressure:"), 1, 0)
        self.pressure_input = QLineEdit("101325")
        self.pressure_input.setPlaceholderText("Pa")
        input_layout.addWidget(self.pressure_input, 1, 1)

        self.pressure_unit = QComboBox()
        self.pressure_unit.addItems(["Pa", "kPa", "bar", "MPa"])
        self.pressure_unit.currentIndexChanged.connect(self._convert_pressure_display)
        input_layout.addWidget(self.pressure_unit, 1, 2)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Engine Selection
        engine_group = self._create_group("Calculation Engine")
        engine_layout = QVBoxLayout()

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["Auto", "CoolProp", "Cantera", "Simplified"])
        engine_layout.addWidget(self.engine_combo)

        # Engine status
        self.engine_status = QLabel()
        self.engine_status.setWordWrap(True)
        self._update_engine_status()
        engine_layout.addWidget(self.engine_status)

        engine_group.setLayout(engine_layout)
        layout.addWidget(engine_group)

        # Calculate button
        calc_btn = QPushButton("Calculate Properties")
        calc_btn.setFont(QFont("Segoe UI", 12, _FONT_WEIGHT_BOLD))
        calc_btn.setMinimumHeight(50)
        calc_btn.clicked.connect(self._calculate)
        layout.addWidget(calc_btn)

        layout.addStretch()
        return panel

    def _create_results_panel(self) -> QWidget:
        """Create the results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        # Results header
        header = QLabel("Calculated Properties")
        header.setFont(QFont("Segoe UI", 14, _FONT_WEIGHT_BOLD))
        header.setStyleSheet(f"color: {COLORS['green']};")
        layout.addWidget(header)

        # Phase indicator
        phase_group = self._create_group("Phase State")
        phase_layout = QHBoxLayout()

        self.phase_label = QLabel("--")
        self.phase_label.setFont(QFont("Segoe UI", 16, _FONT_WEIGHT_BOLD))
        self.phase_label.setAlignment(_ALIGN_CENTER)
        phase_layout.addWidget(self.phase_label)

        self.quality_label = QLabel("Quality: --")
        self.quality_label.setAlignment(_ALIGN_CENTER)
        phase_layout.addWidget(self.quality_label)

        phase_group.setLayout(phase_layout)
        layout.addWidget(phase_group)

        # Thermodynamic Properties
        thermo_group = self._create_group("Thermodynamic Properties")
        thermo_layout = QGridLayout()
        thermo_layout.setSpacing(8)

        properties = [
            ("Temperature", "temp", COLORS["blue"]),
            ("Pressure", "pressure", COLORS["blue"]),
            ("Density", "density", COLORS["peach"]),
            ("Specific Volume", "spec_vol", COLORS["peach"]),
            ("Enthalpy", "enthalpy", COLORS["green"]),
            ("Entropy", "entropy", COLORS["green"]),
            ("Internal Energy", "int_energy", COLORS["teal"]),
            ("Cp", "cp", COLORS["yellow"]),
            ("Cv", "cv", COLORS["yellow"]),
        ]

        for i, (name, key, color) in enumerate(properties):
            row, col = divmod(i, 3)
            card = self._create_result_card(name, key, color)
            thermo_layout.addWidget(card, row, col)

        thermo_group.setLayout(thermo_layout)
        layout.addWidget(thermo_group)

        # Transport Properties
        transport_group = self._create_group("Transport Properties")
        transport_layout = QGridLayout()
        transport_layout.setSpacing(8)

        transport_props = [
            ("Speed of Sound", "sound", COLORS["mauve"]),
            ("Thermal Conductivity", "therm_cond", COLORS["lavender"]),
            ("Dynamic Viscosity", "dyn_visc", COLORS["sky"]),
            ("Kinematic Viscosity", "kin_visc", COLORS["sapphire"]),
        ]

        for i, (name, key, color) in enumerate(transport_props):
            card = self._create_result_card(name, key, color)
            transport_layout.addWidget(card, 0, i)

        transport_group.setLayout(transport_layout)
        layout.addWidget(transport_group)

        # Derived Properties
        derived_group = self._create_group("Derived Properties")
        derived_layout = QGridLayout()
        derived_layout.setSpacing(8)

        derived_props = [
            ("Compressibility (Z)", "compress", COLORS["teal"]),
            ("Prandtl Number", "prandtl", COLORS["peach"]),
            ("Cp/Cv Ratio (k)", "gamma", COLORS["yellow"]),
        ]

        for i, (name, key, color) in enumerate(derived_props):
            card = self._create_result_card(name, key, color)
            derived_layout.addWidget(card, 0, i)

        derived_group.setLayout(derived_layout)
        layout.addWidget(derived_group)

        layout.addStretch()
        return panel

    def _create_group(self, title: str) -> QGroupBox:
        """Create a styled group box.

        Args:
            title: Group box title string.

        Raises:
            TypeError: If title is not a string.
        """
        if not isinstance(title, str):
            raise TypeError(f"title must be a string, got {type(title).__name__}")
        group = QGroupBox(title)
        group.setFont(QFont("Segoe UI", 10, _FONT_WEIGHT_BOLD))
        return group

    def _create_label(self, text: str) -> QLabel:
        """Create a styled label.

        Args:
            text: Label text string.

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        label = QLabel(text)
        label.setStyleSheet(f"color: {COLORS['text']};")
        return label

    def _create_result_card(self, name: str, key: str, color: str) -> QFrame:
        """Create a result display card.

        Args:
            name: Display name for the property.
            key: Dictionary key for the result label.
            color: Hex color string for the value label.

        Raises:
            TypeError: If name, key, or color is not a string.
        """
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}")
        if not isinstance(color, str):
            raise TypeError(f"color must be a string, got {type(color).__name__}")
        card = QFrame()
        card.setFrameShape(_FRAME_STYLED_PANEL)
        card.setMinimumHeight(60)

        layout = QVBoxLayout(card)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        name_label = QLabel(name)
        name_label.setStyleSheet(f"color: {COLORS['subtext0']}; font-size: 10px;")

        value_label = QLabel("--")
        value_label.setFont(QFont("Segoe UI", 11, _FONT_WEIGHT_BOLD))
        value_label.setStyleSheet(f"color: {color};")

        self.result_labels[key] = value_label

        layout.addWidget(name_label)
        layout.addWidget(value_label)

        return card

    def _apply_styles(self) -> None:
        """Apply Catppuccin Mocha theme styles."""
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background-color: {COLORS["base"]};
                color: {COLORS["text"]};
            }}
            QGroupBox {{
                background-color: {COLORS["mantle"]};
                border: 1px solid {COLORS["surface1"]};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {COLORS["lavender"]};
            }}
            QLineEdit {{
                background-color: {COLORS["surface0"]};
                border: 1px solid {COLORS["surface1"]};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS["text"]};
            }}
            QLineEdit:focus {{
                border-color: {COLORS["blue"]};
            }}
            QComboBox {{
                background-color: {COLORS["surface0"]};
                border: 1px solid {COLORS["surface1"]};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS["text"]};
                min-width: 80px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS["surface0"]};
                color: {COLORS["text"]};
                selection-background-color: {COLORS["surface1"]};
            }}
            QPushButton {{
                background-color: {COLORS["blue"]};
                color: {COLORS["base"]};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS["sapphire"]};
            }}
            QPushButton:pressed {{
                background-color: {COLORS["sky"]};
            }}
            QFrame {{
                background-color: {COLORS["surface0"]};
                border-radius: 6px;
            }}
            QScrollArea {{
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {COLORS["mantle"]};
                width: 12px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {COLORS["surface1"]};
                border-radius: 6px;
                min-height: 20px;
            }}
        """
        )

    def _on_mode_changed(self, index: int) -> None:
        """Handle calculation mode change."""
        if index == self.MODE_TP:
            self.temp_input.setEnabled(True)
            self.pressure_input.setEnabled(True)
        elif index == self.MODE_SAT_T:
            self.temp_input.setEnabled(True)
            self.pressure_input.setEnabled(False)
            self.pressure_input.setText("(calculated)")
        elif index == self.MODE_SAT_P:
            self.temp_input.setEnabled(False)
            self.temp_input.setText("(calculated)")
            self.pressure_input.setEnabled(True)

    def _convert_temp_display(self) -> None:
        """Convert temperature display between units."""
        try:
            text = self.temp_input.text()
            if text == "(calculated)":
                return
            value = float(text)
            unit = self.temp_unit.currentText()

            if unit == "°C":
                # Convert K to °C for display
                self.temp_input.setText(f"{value - 273.15:.2f}")
            else:
                # Already in K
                pass
        except ValueError:
            pass

    def _convert_pressure_display(self) -> None:
        """Convert pressure display between units."""
        # Not converting, just for consistency

    def _update_engine_status(self) -> None:
        """Update engine availability status."""
        status_parts = []
        try:
            from upstream_drift_tools.calculators.thermo.steam_engine import (
                CANTERA_AVAILABLE,
                COOLPROP_AVAILABLE,
            )

            if COOLPROP_AVAILABLE:
                status_parts.append("CoolProp: Available")
            else:
                status_parts.append("CoolProp: Not installed")
            if CANTERA_AVAILABLE:
                status_parts.append("Cantera: Available")
            else:
                status_parts.append("Cantera: Not installed")
        except ImportError:
            status_parts.append("Engine: Not available")

        self.engine_status.setText(" | ".join(status_parts))
        self.engine_status.setStyleSheet(
            f"color: {COLORS['subtext0']}; font-size: 10px;"
        )

    def _get_temperature_k(self) -> float:
        """Get temperature in Kelvin from input."""
        text = self.temp_input.text()
        if text == "(calculated)":
            return 0.0
        value = float(text)
        if self.temp_unit.currentText() == "°C":
            return value + 273.15
        return value

    def _get_pressure_pa(self) -> float:
        """Get pressure in Pascals from input."""
        text = self.pressure_input.text()
        if text == "(calculated)":
            return 0.0
        value = float(text)
        unit = self.pressure_unit.currentText()
        if unit == "kPa":
            return value * 1000
        if unit == "bar":
            return value * 100000
        if unit == "MPa":
            return value * 1000000
        return value

    def _calculate(self) -> None:
        """Perform the calculation."""
        if self.engine is None:
            QMessageBox.warning(
                self,
                "Engine Not Available",
                "Steam calculation engine is not available.\nPlease check your installation.",
            )
            return

        try:
            mode = self.mode_combo.currentIndex()
            engine_name = self.engine_combo.currentText().lower()

            if mode == self.MODE_TP:
                temp_k = self._get_temperature_k()
                pressure_pa = self._get_pressure_pa()

                # Validate inputs
                valid, msg = validate_temperature_k(temp_k)
                if not valid:
                    QMessageBox.warning(self, "Invalid Temperature", msg)
                    return

                valid, msg = validate_pressure_pa(pressure_pa)
                if not valid:
                    QMessageBox.warning(self, "Invalid Pressure", msg)
                    return

                result = self.engine.calculate_properties(
                    temperature=temp_k, pressure=pressure_pa, engine=engine_name
                )

            elif mode == self.MODE_SAT_T:
                temp_k = self._get_temperature_k()

                valid, msg = validate_temperature_k(temp_k)
                if not valid:
                    QMessageBox.warning(self, "Invalid Temperature", msg)
                    return

                result = self.engine.calculate_saturated_properties_from_temperature(
                    temperature=temp_k, engine=engine_name
                )

            else:  # MODE_SAT_P
                pressure_pa = self._get_pressure_pa()

                valid, msg = validate_pressure_pa(pressure_pa)
                if not valid:
                    QMessageBox.warning(self, "Invalid Pressure", msg)
                    return

                result = self.engine.calculate_saturated_properties_from_pressure(
                    pressure=pressure_pa, engine=engine_name
                )

            self._display_results(result)

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")
        except (RuntimeError, TypeError, ArithmeticError) as e:
            QMessageBox.critical(self, "Calculation Error", f"Error: {e}")

    def _display_results(self, result: SteamProperties) -> None:
        """Display calculation results.

        Args:
            result: SteamProperties dataclass with calculated values.

        Raises:
            ValueError: If result is None.
        """
        if result is None:
            raise ValueError("result must not be None")
        # Phase state
        phase_colors = {
            "liquid": COLORS["blue"],
            "vapor": COLORS["peach"],
            "two-phase": COLORS["yellow"],
            "supercritical": COLORS["red"],
        }
        phase_color = phase_colors.get(result.phase.lower(), COLORS["text"])
        self.phase_label.setText(result.phase.upper())
        self.phase_label.setStyleSheet(
            f"color: {phase_color}; font-size: 16px; font-weight: bold;"
        )

        quality_text = (
            f"Quality: {result.quality:.4f}"
            if result.quality is not None
            else "Quality: N/A"
        )
        self.quality_label.setText(quality_text)

        # Thermodynamic properties
        self.result_labels["temp"].setText(
            f"{result.temperature:.2f} K ({result.temperature - 273.15:.2f} °C)"
        )
        self.result_labels["pressure"].setText(
            f"{result.pressure / 1000:.2f} kPa ({result.pressure / 1e5:.4f} bar)"
        )
        self.result_labels["density"].setText(f"{result.density:.4f} kg/m³")
        self.result_labels["spec_vol"].setText(f"{result.specific_volume:.6f} m³/kg")
        self.result_labels["enthalpy"].setText(f"{result.enthalpy / 1000:.2f} kJ/kg")
        self.result_labels["entropy"].setText(f"{result.entropy / 1000:.4f} kJ/kg-K")
        self.result_labels["int_energy"].setText(
            f"{result.internal_energy / 1000:.2f} kJ/kg"
        )
        self.result_labels["cp"].setText(f"{result.cp:.2f} J/kg-K")
        self.result_labels["cv"].setText(f"{result.cv:.2f} J/kg-K")

        # Transport properties
        self.result_labels["sound"].setText(f"{result.speed_of_sound:.2f} m/s")
        self.result_labels["therm_cond"].setText(
            f"{result.thermal_conductivity:.6f} W/m-K"
        )
        self.result_labels["dyn_visc"].setText(f"{result.dynamic_viscosity:.2e} Pa·s")
        self.result_labels["kin_visc"].setText(f"{result.kinematic_viscosity:.2e} m²/s")

        # Derived properties
        if result.compressibility_factor is not None:
            self.result_labels["compress"].setText(
                f"{result.compressibility_factor:.4f}"
            )
        else:
            self.result_labels["compress"].setText("--")

        if result.prandtl_number is not None:
            self.result_labels["prandtl"].setText(f"{result.prandtl_number:.4f}")
        else:
            self.result_labels["prandtl"].setText("--")

        if result.specific_heat_ratio is not None:
            self.result_labels["gamma"].setText(f"{result.specific_heat_ratio:.4f}")
        else:
            self.result_labels["gamma"].setText("--")


def main() -> None:
    """Main entry point for standalone execution."""
    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    window = SteamEngineCalculatorWindow()
    setup_themed_app(app, window, settings_app="SteamEngineCalculator")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
