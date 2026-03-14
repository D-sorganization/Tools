"""EventsMixin -- timers, input handlers, mouse interaction, view presets."""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QTimer, pyqtSlot
from PyQt6.QtWidgets import QMessageBox

from ...configs.view_presets import (
    DEFAULT_Z_SCALE_FACTOR,
    DEFAULT_ZOOM_SCALE_FACTOR,
    get_view_preset,
)

logger = logging.getLogger(__name__)


class EventsMixin:
    """Mixin providing event handlers, timers, and mouse interaction."""

    # -- Attributes provided by the host class (declared for mypy) --
    auto_scale_checkbox: Any
    bath_diameter_input: Any
    connect_glass_calculator_btn: Any
    depth_inputs: Any
    electrode_ax: Any
    electrode_canvas: Any
    electrode_extension_slider: Any
    electrode_extension_value_label: Any
    glass_integration_checkbox: Any
    glass_layer_height_input: Any
    max_scale_input: Any
    metal_conductive_checkbox: Any
    metal_layer_height_input: Any
    min_scale_input: Any
    optimization_complete: Any
    rotation_mode_radio: Any
    zoom_label: Any
    zoom_slider: Any
    _calculate_system: Any
    _check_glass_calculator_availability: Any
    _update_analysis_display: Any
    _update_glass_integration_status: Any
    _update_results_tables: Any

    def _setup_timers(self) -> None:
        """Setup update timers (periodic update removed — was a no-op)."""
        self.calc_timer = QTimer()

    @pyqtSlot()
    def _on_input_changed(self) -> None:
        """Handle input parameter changes"""
        if getattr(self, "_initialization_complete", False):
            # Only call _calculate_system() which internally calls _update_3d_visualization()
            # Remove the duplicate _draw_3d_real_geometry() call that was causing conflicts
            self._calculate_system()

    @pyqtSlot()
    def _on_electrode_extension_changed(self) -> None:
        """Update label and trigger input change when slider moves"""
        value = self.electrode_extension_slider.value()
        self.electrode_extension_value_label.setText(f"{value} in")
        self._on_input_changed()

    @pyqtSlot()
    def _on_auto_scale_changed(self) -> None:
        """Handle auto-scale checkbox state change"""
        state = self.auto_scale_checkbox.isChecked()
        if self.min_scale_input and self.max_scale_input:
            self.min_scale_input.setEnabled(not state)
            self.max_scale_input.setEnabled(not state)

    @pyqtSlot()
    def _on_zoom_slider_changed(self) -> None:
        """Handle zoom slider changes"""
        zoom_value = self.zoom_slider.value()
        self.zoom_label.setText(f"Zoom: {zoom_value}%")

        # Apply zoom to 3D plot
        if self.electrode_ax is not None:
            # Get current view limits
            xlim = self.electrode_ax.get_xlim()
            ylim = self.electrode_ax.get_ylim()

            # Calculate center
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2

            # Calculate base range (at 100% zoom)
            base_range = (
                self.bath_diameter_input.value() / 2
                + self.electrode_extension_slider.value()
            )

            # Apply zoom factor
            zoom_factor = zoom_value / 100.0
            new_range = base_range / zoom_factor * 1.1  # 1.1 for margin

            # Set new limits
            self.electrode_ax.set_xlim(
                x_center - new_range / 2, x_center + new_range / 2
            )
            self.electrode_ax.set_ylim(
                y_center - new_range / 2, y_center + new_range / 2
            )
            if hasattr(self.electrode_ax, "set_zlim"):
                z_range = (
                    (
                        self.glass_layer_height_input.value()
                        + self.metal_layer_height_input.value()
                    )
                    / zoom_factor
                    * 1.2
                )
                self.electrode_ax.set_zlim(0, z_range)

            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()

    @pyqtSlot()
    def _validate_glass_height(self) -> None:
        """Validate that glass height is above electrode tips"""
        try:
            glass_height = self.glass_layer_height_input.value()
            metal_height = self.metal_layer_height_input.value()

            # Get maximum electrode depth
            max_electrode_depth: float = 0.0
            for i in range(3):
                if i in self.depth_inputs:
                    depth = self.depth_inputs[i].value()
                    max_electrode_depth = max(max_electrode_depth, depth)

            # Calculate minimum glass height needed (should be above electrode tips)
            min_glass_height = (
                max_electrode_depth + metal_height + 1.0
            )  # 1 inch safety margin

            if glass_height < min_glass_height:
                # Show warning and suggest minimum height
                QMessageBox.warning(
                    self,  # type: ignore[arg-type]
                    "Glass Height Warning",
                    f"Glass height ({glass_height:.1f} in) should be above electrode tips.\n"
                    f"Minimum recommended height: {min_glass_height:.1f} in\n"
                    f"(Electrode depth: {max_electrode_depth:.1f} in + "
                    f"Metal height: {metal_height:.1f} in + Safety margin: 1.0 in)",
                )

                # Automatically adjust to minimum safe height
                self.glass_layer_height_input.setValue(min_glass_height)

            # Continue with normal calculation
            self._calculate_system()

        except (RuntimeError, AttributeError) as e:
            logger.exception("Error in glass height validation: %s", e)
            # Fall back to normal calculation
            self._calculate_system()

    @pyqtSlot()
    def _on_metal_conductivity_changed(self) -> None:
        """Handle metal layer conductivity toggle"""
        try:
            is_enabled = self.metal_conductive_checkbox.isChecked()
            logger.debug(
                "[DEBUG] Metal layer conductivity: %s",
                "Enabled" if is_enabled else "Disabled",
            )

            # If metal conductivity is disabled, hide metal layer in visualization
            if hasattr(self, "show_metal_checkbox"):
                if not is_enabled:
                    # Disable metal layer visualization when conductivity is off
                    self.show_metal_checkbox.setChecked(False)
                    # Optionally disable the checkbox to prevent manual enabling
                    self.show_metal_checkbox.setEnabled(False)
                    # Add visual indication that it's disabled
                    self.show_metal_checkbox.setStyleSheet(
                        "QCheckBox { color: #888888; }"
                    )
                    self.show_metal_checkbox.setToolTip(
                        "Metal layer visualization disabled when conduction is off"
                    )
                else:
                    # Re-enable metal layer visualization controls
                    self.show_metal_checkbox.setEnabled(True)
                    self.show_metal_checkbox.setChecked(True)
                    self.show_metal_checkbox.setStyleSheet("")
                    self.show_metal_checkbox.setToolTip("")

            # Recalculate system with new conductivity model
            self._calculate_system()

            # Update results tables to reflect new percentages
            self._update_results_tables()
            self._update_analysis_display()

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error handling metal conductivity change: %s", e)
            import traceback

            traceback.print_exc()

    def _on_interaction_mode_changed(self) -> None:
        """Handle interaction mode change"""
        try:
            if self.rotation_mode_radio.isChecked():
                self.interaction_mode = "rotation"
            else:
                self.interaction_mode = "pan"
            logger.info("3D interaction mode changed to: %s", self.interaction_mode)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error changing interaction mode: %s", e)

    def _on_scroll(self, event: Any) -> None:
        """Handle mouse scroll for zoom"""
        try:
            if event.inaxes == self.electrode_ax:
                # Zoom factor
                zoom_factor = 1.1 if event.button == "down" else 0.9

                # Update zoom slider
                current_zoom = self.zoom_slider.value()
                new_zoom = int(current_zoom * zoom_factor)
                new_zoom = max(50, min(200, new_zoom))  # Clamp to range
                self.zoom_slider.setValue(new_zoom)

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error in scroll handler: %s", e)

    def _on_mouse_press(self, event: Any) -> None:
        """Handle mouse press for rotation and pan"""
        try:
            if event.inaxes == self.electrode_ax:
                self._mouse_pressed = True
                # Use pixel coordinates for consistent behavior
                self._last_mouse_pos = (
                    (event.x, event.y)
                    if event.x is not None and event.y is not None
                    else None
                )
                # Get 3D view angles for rotation mode
                if hasattr(self.electrode_ax, "elev") and hasattr(
                    self.electrode_ax, "azim"
                ):
                    self._last_elev = self.electrode_ax.elev
                    self._last_azim = self.electrode_ax.azim
                else:
                    # Fallback: use default angles if not available
                    self._last_elev = 20
                    self._last_azim = 45
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error in mouse press handler: %s", e)

    def _on_mouse_release(self, event: Any) -> None:
        """Handle mouse release"""
        try:
            self._mouse_pressed = False
            logger.info("Mouse released")
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error in mouse release handler: %s", e)

    def _on_mouse_motion(self, event: Any) -> None:
        """Handle mouse motion for rotation or panning based on interaction mode"""
        try:
            if (
                self._mouse_pressed
                and event.inaxes == self.electrode_ax
                and self._last_mouse_pos
                and event.x is not None
                and event.y is not None
            ):
                dx = event.x - self._last_mouse_pos[0]
                dy = event.y - self._last_mouse_pos[1]
                logger.info(
                    "Mouse motion: dx=%s, dy=%s, mode=%s", dx, dy, self.interaction_mode
                )
                if self.interaction_mode == "rotation":
                    # Only rotate if in rotation mode
                    if hasattr(self.electrode_ax, "view_init") and hasattr(
                        self, "_last_elev"
                    ):
                        rotation_sensitivity = 0.5
                        new_elev = self._last_elev + dy * rotation_sensitivity
                        new_azim = self._last_azim + dx * rotation_sensitivity
                        new_elev = max(-90, min(90, new_elev))
                        self.electrode_ax.view_init(elev=new_elev, azim=new_azim)
                        if self.electrode_canvas:
                            self.electrode_canvas.draw_idle()
                        self._last_elev = new_elev
                        self._last_azim = new_azim
                elif self.interaction_mode == "pan":
                    # Only pan if in pan mode
                    if hasattr(self.electrode_ax, "get_xlim"):
                        x_range = self.electrode_ax.get_xlim()
                        y_range = self.electrode_ax.get_ylim()
                        pan_factor = 0.01
                        x_shift = dx * pan_factor * (x_range[1] - x_range[0])
                        y_shift = dy * pan_factor * (y_range[1] - y_range[0])
                        self.electrode_ax.set_xlim(
                            x_range[0] - x_shift, x_range[1] - x_shift
                        )
                        self.electrode_ax.set_ylim(
                            y_range[0] + y_shift, y_range[1] + y_shift
                        )
                    if self.electrode_canvas is not None:
                        self.electrode_canvas.draw_idle()
                self._last_mouse_pos = (event.x, event.y)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error in mouse motion handler: %s", e)

    def _reset_3d_view(self) -> None:
        """Reset 3D view to default angle and zoom"""
        try:
            if self.electrode_canvas is None or self.electrode_ax is None:
                return

            if hasattr(self.electrode_ax, "view_init"):
                self.electrode_ax.view_init(elev=20, azim=45)

            # Reset zoom slider to 100%
            if hasattr(self, "zoom_slider"):
                self.zoom_slider.setValue(100)

            self.electrode_canvas.draw_idle()
            logger.info("3D view reset to default")
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error resetting 3D view: %s", e)

    def _set_view_preset(self, preset: str) -> None:
        """Set predefined view angles using view presets from config."""
        if self.electrode_ax is None:
            return

        try:
            view_angle = get_view_preset(preset)
            self.electrode_ax.view_init(elev=view_angle.elev, azim=view_angle.azim)

            # For default view, also reset pan/zoom
            if preset == "default":
                bath_diameter = self.bath_diameter_input.value()
                extension_length = float(self.electrode_extension_slider.value())
                glass_height = self.glass_layer_height_input.value()
                metal_height = self.metal_layer_height_input.value()
                max_range = max(
                    bath_diameter / 2 + extension_length, glass_height + metal_height
                )
                zoom_factor = self.zoom_slider.value() / 100.0
                scaled_range = max_range / zoom_factor * DEFAULT_ZOOM_SCALE_FACTOR
                self.electrode_ax.set_xlim(-scaled_range, scaled_range)
                self.electrode_ax.set_ylim(-scaled_range, scaled_range)
                if hasattr(self.electrode_ax, "set_zlim"):
                    self.electrode_ax.set_zlim(
                        0,
                        (glass_height + metal_height)
                        / zoom_factor
                        * DEFAULT_Z_SCALE_FACTOR,
                    )

            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error setting view preset: %s", e)

    def _on_color_scheme_changed(self, scheme: str) -> None:
        """Handle color scheme changes"""
        try:
            self.current_color_scheme = scheme
            # Trigger visualization update with new color scheme
            self._on_input_changed()
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error changing color scheme: %s", e)

    def _on_glass_integration_changed(self, state: int) -> None:
        """Handle glass integration checkbox state change"""
        try:
            if state == 2:  # Qt.CheckState.Checked
                # Check if glass calculator is available
                if self._check_glass_calculator_availability():
                    self.connect_glass_calculator_btn.setEnabled(True)
                    self._update_glass_integration_status("Available", "ok")
                else:
                    self.glass_integration_checkbox.setChecked(False)
                    self._update_glass_integration_status("Not Available", "error")
                    QMessageBox.warning(
                        self,  # type: ignore[arg-type]
                        "Glass Properties Calculator Not Available",
                        "The Glass Properties Calculator is not available "
                        "or not properly loaded.\n\n"
                        "Please ensure the Glass Properties tab is available\n"
                        "in the main application.",
                    )
            else:
                self.connect_glass_calculator_btn.setEnabled(False)
                self._update_glass_integration_status("Disabled", "neutral")

        except (RuntimeError, AttributeError) as e:
            logger.exception("Error in glass integration change handler: %s", e)

    @pyqtSlot()
    def _run_optimization(self) -> None:
        """Run electrode position optimization"""
        # This would implement the optimization algorithm
        # For now, just show a message
        QMessageBox.information(
            self,  # type: ignore[arg-type]
            "Optimization",
            "Optimization feature will be implemented with full algorithm integration",
        )

        # Emit optimization complete signal
        self.optimization_complete.emit({"status": "pending"})
