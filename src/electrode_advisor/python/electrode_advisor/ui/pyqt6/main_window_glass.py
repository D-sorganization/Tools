"""GlassIntegrationMixin -- glass properties calculator integration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QMessageBox

from ...configs.color_schemes import GLASS_INTEGRATION_COLORS

logger = logging.getLogger(__name__)


class GlassIntegrationMixin:
    """Mixin providing glass properties calculator integration.

    This mixin is designed to be used with the main ElectrodeAdvisor window.
    It expects the host class to provide: glass_interface, _calculate_system(),
    and _update_status().
    """

    # Declare expected attributes from the host class for type checking
    glass_interface: Any
    _calculate_system: Callable[..., Any]
    _update_status: Callable[..., Any]

    def set_glass_properties_calculator(self, calculator: Callable[..., Any]) -> None:
        """Set external glass properties calculator"""
        self.glass_interface.set_external_calculator(calculator)
        self._calculate_system()

    def connect_to_glass_calculator(self, glass_calculator_widget: Any) -> bool:
        """Connect to a glass properties calculator widget for resistivity predictions"""
        try:
            if hasattr(glass_calculator_widget, "glass_calculator"):
                # The widget is a wrapper, get the actual calculator
                actual_calculator = glass_calculator_widget.glass_calculator

                # Create a callback function that interfaces with the glass calculator
                def glass_property_callback(
                    temperature: float,
                    composition: Any = None,
                    power_density: float = 0,
                ) -> float:
                    """Glass Property Callback method.

                    Returns:
                        Glass conductivity value
                    """
                    if hasattr(actual_calculator, "calculate_resistivity"):
                        # Get resistivity and convert to conductivity
                        resistivity = actual_calculator.calculate_resistivity(
                            temperature, composition
                        )
                        return float(1.0 / resistivity if resistivity > 0 else 0.0)
                    if hasattr(actual_calculator, "calculate_conductivity"):
                        return float(
                            actual_calculator.calculate_conductivity(
                                temperature, composition
                            )
                        )
                    # Fallback to default model
                    return float(
                        self.glass_interface._default_conductivity_model(
                            temperature, power_density
                        )
                    )

                # Set the callback
                self.glass_interface.set_external_calculator(glass_property_callback)

                # Connect any signals if available
                if hasattr(actual_calculator, "properties_changed"):
                    actual_calculator.properties_changed.connect(
                        lambda: self._calculate_system()
                    )

                logger.info("Successfully connected to glass properties calculator")
                return True
            return False

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error connecting to glass calculator: %s", e)
            return False

    def _check_glass_calculator_availability(self) -> bool:
        """Check if glass calculator is available in the main application"""
        try:
            # Try to find the main window and check for glass calculator
            main_window = self._find_main_window()
            if not main_window:
                return False

            # Check for glass calculator in TabManager (preferred)
            if (
                hasattr(main_window, "tab_manager")
                and hasattr(main_window.tab_manager, "glass_calculator_tab")
                and main_window.tab_manager.glass_calculator_tab is not None
            ):
                return True

            # Legacy check (fallback)
            return bool(
                hasattr(main_window, "glass_calculator_tab")
                and main_window.glass_calculator_tab
            )
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error checking glass calculator availability: %s", e)
            return False

    def _find_main_window(self) -> QObject | None:
        """Find the main application window"""
        try:
            # Walk up the widget hierarchy to find the main window
            widget: QObject | None = self  # type: ignore[assignment]
            while widget and not hasattr(widget, "glass_calculator_tab"):
                widget = widget.parent()
            return widget
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error finding main window: %s", e)
            return None

    def _connect_to_glass_calculator(self) -> None:
        """Connect to the glass calculator"""
        try:
            main_window = self._find_main_window()
            glass_calculator = None

            if main_window:
                # Try TabManager first
                if hasattr(main_window, "tab_manager") and hasattr(
                    main_window.tab_manager, "glass_calculator_tab"
                ):
                    glass_calculator = main_window.tab_manager.glass_calculator_tab

                # Fallback to direct attribute
                if glass_calculator is None and hasattr(
                    main_window, "glass_calculator_tab"
                ):
                    glass_calculator = main_window.glass_calculator_tab

            if glass_calculator:
                self.connect_to_glass_calculator(glass_calculator)

                # Update the glass interface with the calculator
                if hasattr(self.glass_interface, "set_external_calculator"):
                    self.glass_interface.set_external_calculator(glass_calculator)

                self._update_status(
                    "Successfully connected to Glass Properties Calculator", "ok"
                )
            else:
                self._update_glass_integration_status("Not Found", "error")
                QMessageBox.warning(
                    self,  # type: ignore[arg-type]
                    "Glass Calculator Not Found",
                    "Could not find the Glass Properties Calculator in the main application.\n\n"
                    "Please ensure the Glass Properties tab is loaded.",
                )

        except (RuntimeError, AttributeError) as e:
            self._update_glass_integration_status("Connection Error", "error")
            self._update_status(
                f"Failed to connect to glass calculator: {e!s}", "error"
            )

    def _update_glass_integration_status(
        self, status_text: str, status_type: str
    ) -> None:
        """Update the glass integration status display"""
        try:
            if hasattr(self, "glass_integration_status"):
                color = GLASS_INTEGRATION_COLORS.get(status_type, "#666666")
                self.glass_integration_status.setText(
                    f"Glass Properties Calculator: {status_text}"
                )
                self.glass_integration_status.setStyleSheet(
                    f"QLabel {{ color: {color}; font-size: 9pt; }}"
                )
        except (RuntimeError, AttributeError) as e:
            logger.exception("Error updating glass integration status: %s", e)

    def _on_glass_properties_updated(self, properties: dict[str, Any]) -> None:
        """Handle glass properties updates from external calculator"""
        try:
            # Update the glass interface
            if hasattr(self.glass_interface, "update_properties"):
                self.glass_interface.update_properties(properties)

            # Update the display
            if hasattr(self, "glass_properties_display"):
                display_text = "Current Glass Properties:\n"
                for key, value in properties.items():
                    if isinstance(value, int | float):
                        display_text += f"  {key}: {value:.3f}\n"
                    else:
                        display_text += f"  {key}: {value}\n"
                self.glass_properties_display.setText(display_text)

            # Recalculate if we have current results
            if hasattr(self, "calculation_results") and self.calculation_results:
                self._calculate_system()

            self._update_status(
                "Glass properties updated and system recalculated", "ok"
            )

        except (RuntimeError, AttributeError) as e:
            self._update_status(f"Glass properties update failed: {e!s}", "error")
