"""PyQt6 widget for mesh quality visualization and analysis.

Provides UI controls for:
- Quality metric selector (Aspect Ratio, Skewness, Jacobian)
- Colormap visualization of per-element metrics
- Statistics panel (min, max, mean, std)
- Problem element counter and identification
- Hover tooltips with per-element metrics
"""

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class MeshQualityWidget(QWidget):
    """Widget for mesh quality analysis and visualization.

    Displays mesh quality metrics with color-coded visualization:
    - Aspect Ratio: dimension ratio (red=poor, green=good)
    - Skewness: element distortion (red=0.0 good, dark=1.0 degenerate)
    - Jacobian: element orientation (red=negative, green=positive)

    Signals:
        metric_changed: Emitted when quality metric changes (str)
        colormap_changed: Emitted when colormap selection changes (str)
        element_selected: Emitted when element is selected (int)
        threshold_changed: Emitted when problem threshold changes (float)
    """

    metric_changed = pyqtSignal(str)
    colormap_changed = pyqtSignal(str)
    element_selected = pyqtSignal(int)
    threshold_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize mesh quality widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.mesh_analyzer: Any = None
        self.current_metric: str = "skewness"
        self.current_colormap: str = "viridis"
        self.problematic_threshold: float = 0.1

        logger.debug("MeshQualityWidget initialized")

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        main_layout = QVBoxLayout(self)

        # Metric selector
        metric_group = QGroupBox("Quality Metric", self)
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Metric:"))

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Skewness", "Aspect Ratio", "Jacobian"])
        metric_layout.addWidget(self.metric_combo)

        metric_group.setLayout(metric_layout)
        main_layout.addWidget(metric_group)

        # Colormap selector
        colormap_group = QGroupBox("Colormap", self)
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            ["viridis", "plasma", "inferno", "magma", "RdYlGn_r", "cool"]
        )
        colormap_layout.addWidget(self.colormap_combo)

        colormap_group.setLayout(colormap_layout)
        main_layout.addWidget(colormap_group)

        # Statistics panel
        stats_group = QGroupBox("Statistics", self)
        stats_layout = QVBoxLayout()

        # Create table for statistics
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setRowCount(4)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.resizeColumnsToContents()

        # Add stat rows
        stat_names = ["Min", "Max", "Mean", "Std Dev"]
        for i, name in enumerate(stat_names):
            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            self.stats_table.setItem(i, 1, QTableWidgetItem("--"))

        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)

        # Problem elements panel
        problem_group = QGroupBox("Problem Elements", self)
        problem_layout = QVBoxLayout()

        # Problem threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(50)  # 0-50% threshold
        self.threshold_slider.setValue(10)  # Default 10%
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel("10%")
        threshold_layout.addWidget(self.threshold_label)

        problem_layout.addLayout(threshold_layout)

        # Problem element counter
        counter_layout = QHBoxLayout()
        counter_layout.addWidget(QLabel("Problem Elements:"))

        self.problem_counter = QLabel("0 / 0")
        self.problem_counter.setStyleSheet("font-weight: bold; color: red;")
        counter_layout.addWidget(self.problem_counter)
        counter_layout.addStretch()

        problem_layout.addLayout(counter_layout)

        # Problem element list
        self.problem_list_label = QLabel("Elements with issues:")
        problem_layout.addWidget(self.problem_list_label)

        self.problem_list = QTableWidget()
        self.problem_list.setColumnCount(3)
        self.problem_list.setHorizontalHeaderLabels(["Element ID", "Metric", "Value"])
        self.problem_list.setMaximumHeight(150)
        problem_layout.addWidget(self.problem_list)

        problem_group.setLayout(problem_layout)
        main_layout.addWidget(problem_group)

        # Action buttons
        action_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Analysis")
        action_layout.addWidget(self.refresh_button)

        self.export_button = QPushButton("Export Metrics")
        action_layout.addWidget(self.export_button)

        action_layout.addStretch()
        main_layout.addLayout(action_layout)

        main_layout.addStretch()

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.refresh_button.clicked.connect(self._on_refresh)
        self.export_button.clicked.connect(self._on_export)

    def _on_metric_changed(self, text: str) -> None:
        """Handle metric selection change."""
        metric_map = {
            "Skewness": "skewness",
            "Aspect Ratio": "aspect_ratio",
            "Jacobian": "jacobian",
        }
        self.current_metric = metric_map.get(text, "skewness")
        self.metric_changed.emit(self.current_metric)
        self._update_display()

    def _on_colormap_changed(self, text: str) -> None:
        """Handle colormap selection change."""
        self.current_colormap = text
        self.colormap_changed.emit(self.current_colormap)
        self._update_display()

    def _on_threshold_changed(self, value: int) -> None:
        """Handle threshold slider change."""
        self.problematic_threshold = value / 100.0
        self.threshold_label.setText(f"{value}%")
        self.threshold_changed.emit(self.problematic_threshold)
        self._update_problem_elements()

    def _on_refresh(self) -> None:
        """Handle refresh button click."""
        logger.debug("Refresh analysis triggered")
        self._update_display()

    def _on_export(self) -> None:
        """Handle export button click."""
        logger.debug("Export metrics triggered")
        # Export functionality would be implemented here

    def set_analyzer(self, analyzer: Any) -> None:
        """Set the mesh quality analyzer.

        Args:
            analyzer: MeshQualityAnalyzer instance
        """
        self.mesh_analyzer = analyzer
        logger.debug("Analyzer set in quality widget")
        self._update_display()

    def _update_display(self) -> None:
        """Update all display elements."""
        if self.mesh_analyzer is None:
            logger.debug("No analyzer set, skipping display update")
            return

        self._update_statistics()
        self._update_problem_elements()

    def _update_statistics(self) -> None:
        """Update statistics panel."""
        if self.mesh_analyzer is None:
            return

        try:
            stats = self.mesh_analyzer.get_statistics(self.current_metric)

            stat_values = [
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
            ]

            for i, value in enumerate(stat_values):
                item = self.stats_table.item(i, 1)
                if item:
                    item.setText(value)

            logger.debug("Statistics updated for metric: %s", self.current_metric)

        except Exception as e:
            logger.error("Error updating statistics: %s", e)

    def _update_problem_elements(self) -> None:
        """Update problem elements list."""
        if self.mesh_analyzer is None:
            self.problem_counter.setText("0 / 0")
            self.problem_list.setRowCount(0)
            return

        try:
            problematic = self.mesh_analyzer.get_problematic_elements(
                threshold=self.problematic_threshold
            )

            # Update counter
            total = len(self.mesh_analyzer.elements)
            self.problem_counter.setText(f"{len(problematic)} / {total}")

            if len(problematic) > 0:
                self.problem_counter.setStyleSheet("font-weight: bold; color: red;")
            else:
                self.problem_counter.setStyleSheet("font-weight: bold; color: green;")

            # Update problem list
            self.problem_list.setRowCount(0)

            if self.current_metric == "skewness":
                metrics = self.mesh_analyzer.compute_skewness()
            elif self.current_metric == "aspect_ratio":
                metrics = self.mesh_analyzer.compute_aspect_ratios()
            elif self.current_metric == "jacobian":
                metrics = self.mesh_analyzer.compute_jacobian()
            else:
                metrics = np.array([])

            # Show top problematic elements (up to 10)
            self.problem_list.setRowCount(min(10, len(problematic)))

            for row, elem_idx in enumerate(problematic[:10]):
                metric_value = float(metrics[elem_idx])

                # Element ID
                id_item = QTableWidgetItem(str(elem_idx))
                self.problem_list.setItem(row, 0, id_item)

                # Metric name
                metric_item = QTableWidgetItem(self.current_metric)
                self.problem_list.setItem(row, 1, metric_item)

                # Metric value
                value_item = QTableWidgetItem(f"{metric_value:.6f}")
                self.problem_list.setItem(row, 2, value_item)

                # Highlight poor elements
                if metric_value > 0.5:
                    for col in range(3):
                        item = self.problem_list.item(row, col)
                        if item:
                            item.setBackground(QColor(255, 200, 200))

            logger.debug("Problem elements updated: %d problematic", len(problematic))

        except Exception as e:
            logger.error("Error updating problem elements: %s", e)

    def get_metric_values(self) -> np.ndarray | None:
        """Get current metric values for visualization.

        Returns:
            Array of metric values (one per element) or None
        """
        if self.mesh_analyzer is None:
            return None

        try:
            if self.current_metric == "skewness":
                return self.mesh_analyzer.compute_skewness()
            elif self.current_metric == "aspect_ratio":
                return self.mesh_analyzer.compute_aspect_ratios()
            elif self.current_metric == "jacobian":
                return self.mesh_analyzer.compute_jacobian()
            else:
                return None
        except Exception as e:
            logger.error("Error getting metric values: %s", e)
            return None

    def get_colormap_name(self) -> str:
        """Get current colormap name.

        Returns:
            Colormap name (e.g., 'viridis')
        """
        return self.current_colormap
