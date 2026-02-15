"""PyQt6 Chart Style Panel Widget.

Provides per-series chart style controls including:
- Display mode (line, scatter, line+scatter)
- Line/marker customization
- Trendline configuration
- Axis and legend controls
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from PyQt6.QtWidgets import (
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)


if PYQT6_AVAILABLE:

    class ChartStylePanel(QWidget):
        """Panel for per-series chart style controls."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            # Display mode
            mode_group = QGroupBox("Display Mode")
            mode_layout = QFormLayout(mode_group)

            self._mode_combo = QComboBox()
            self._mode_combo.addItems(["line", "scatter", "line+scatter"])
            mode_layout.addRow("Mode:", self._mode_combo)

            self._line_style_combo = QComboBox()
            self._line_style_combo.addItems(
                [
                    "solid",
                    "dashed",
                    "dotted",
                    "dashdot",
                ]
            )
            mode_layout.addRow("Line Style:", self._line_style_combo)

            self._line_width_spin = QDoubleSpinBox()
            self._line_width_spin.setRange(0.5, 5.0)
            self._line_width_spin.setValue(1.5)
            self._line_width_spin.setSingleStep(0.5)
            mode_layout.addRow("Line Width:", self._line_width_spin)

            self._marker_combo = QComboBox()
            self._marker_combo.addItems(
                [
                    "none",
                    "circle",
                    "square",
                    "triangle",
                    "diamond",
                    "cross",
                    "plus",
                    "star",
                ]
            )
            mode_layout.addRow("Marker:", self._marker_combo)

            self._marker_size_spin = QDoubleSpinBox()
            self._marker_size_spin.setRange(1.0, 20.0)
            self._marker_size_spin.setValue(6.0)
            mode_layout.addRow("Marker Size:", self._marker_size_spin)

            self._opacity_spin = QDoubleSpinBox()
            self._opacity_spin.setRange(0.0, 1.0)
            self._opacity_spin.setValue(1.0)
            self._opacity_spin.setSingleStep(0.1)
            mode_layout.addRow("Opacity:", self._opacity_spin)

            self._color_btn = QPushButton("Pick Color")
            self._color_btn.clicked.connect(self._pick_color)
            self._selected_color: str | None = None
            mode_layout.addRow("Color:", self._color_btn)

            layout.addWidget(mode_group)

            # Trendline
            trend_group = QGroupBox("Trendline")
            trend_layout = QFormLayout(trend_group)

            self._trend_type_combo = QComboBox()
            self._trend_type_combo.addItems(
                [
                    "None",
                    "linear",
                    "polynomial",
                    "exponential",
                    "power",
                ]
            )
            trend_layout.addRow("Type:", self._trend_type_combo)

            self._trend_degree_spin = QSpinBox()
            self._trend_degree_spin.setRange(2, 10)
            self._trend_degree_spin.setValue(2)
            trend_layout.addRow("Poly Degree:", self._trend_degree_spin)

            self._show_equation_check = QCheckBox("Show Equation")
            self._show_equation_check.setChecked(True)
            trend_layout.addRow("", self._show_equation_check)

            self._show_r2_check = QCheckBox("Show R\u00b2")
            self._show_r2_check.setChecked(True)
            trend_layout.addRow("", self._show_r2_check)

            layout.addWidget(trend_group)

            # Axis controls
            axis_group = QGroupBox("Axes")
            axis_layout = QFormLayout(axis_group)

            self._x_label_edit = QComboBox()
            self._x_label_edit.setEditable(True)
            axis_layout.addRow("X Label:", self._x_label_edit)

            self._y_label_edit = QComboBox()
            self._y_label_edit.setEditable(True)
            axis_layout.addRow("Y Label:", self._y_label_edit)

            self._x_log_check = QCheckBox("Log Scale X")
            axis_layout.addRow("", self._x_log_check)

            self._y_log_check = QCheckBox("Log Scale Y")
            axis_layout.addRow("", self._y_log_check)

            self._grid_check = QCheckBox("Show Grid")
            self._grid_check.setChecked(True)
            axis_layout.addRow("", self._grid_check)

            layout.addWidget(axis_group)

            # Legend
            legend_group = QGroupBox("Legend")
            legend_layout = QFormLayout(legend_group)

            self._legend_visible_check = QCheckBox("Show Legend")
            self._legend_visible_check.setChecked(True)
            legend_layout.addRow("", self._legend_visible_check)

            self._legend_pos_combo = QComboBox()
            self._legend_pos_combo.addItems(
                [
                    "right",
                    "left",
                    "top",
                    "bottom",
                    "none",
                ]
            )
            legend_layout.addRow("Position:", self._legend_pos_combo)

            layout.addWidget(legend_group)
            layout.addStretch()

        def _pick_color(self) -> None:
            color = QColorDialog.getColor()
            if color.isValid():
                self._selected_color = color.name()
                self._color_btn.setStyleSheet(
                    f"background-color: {self._selected_color};"
                )

        def get_series_style(self) -> Any:
            """Build a SeriesStyle from current widget state."""
            from plot_engine.specs import SeriesStyle

            return SeriesStyle(
                color=self._selected_color,
                line_style=self._line_style_combo.currentText(),
                line_width=self._line_width_spin.value(),
                marker=self._marker_combo.currentText(),
                marker_size=self._marker_size_spin.value(),
                opacity=self._opacity_spin.value(),
                display_mode=self._mode_combo.currentText(),
            )

        def get_trendline_spec(self) -> Any:
            """Build a TrendlineSpec or None."""
            from plot_engine.specs import TrendlineSpec

            trend_type = self._trend_type_combo.currentText()
            if trend_type == "None":
                return None
            return TrendlineSpec(
                type=trend_type,
                degree=self._trend_degree_spin.value(),
                show_equation=self._show_equation_check.isChecked(),
                show_r_squared=self._show_r2_check.isChecked(),
            )

        def get_axis_specs(self) -> tuple[Any, Any]:
            """Build X and Y AxisSpec from current widget state."""
            from plot_engine.specs import AxisSpec

            x_axis = AxisSpec(
                label=self._x_label_edit.currentText(),
                log_scale=self._x_log_check.isChecked(),
                grid=self._grid_check.isChecked(),
            )
            y_axis = AxisSpec(
                label=self._y_label_edit.currentText(),
                log_scale=self._y_log_check.isChecked(),
                grid=self._grid_check.isChecked(),
            )
            return x_axis, y_axis

        def get_legend_spec(self) -> Any:
            """Build a LegendSpec from current widget state."""
            from plot_engine.specs import LegendSpec

            return LegendSpec(
                visible=self._legend_visible_check.isChecked(),
                position=self._legend_pos_combo.currentText(),
            )
