"""PyQt6 Dialog Widgets for Plot Visualizations.

Provides dialog components for:
- Contour Plot creation
- Heatmap generation
- Filter comparison visualization
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    QDialog = object  # type: ignore[misc,assignment]
    QWidget = object  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)


if PYQT6_AVAILABLE:

    class ContourPlotDialog(QDialog):
        """Dialog for creating contour plots from DataFrame columns."""

        def __init__(
            self,
            df: pd.DataFrame,
            parent: QWidget | None = None,
        ) -> None:
            assert df is not None, "df must be provided"
            super().__init__(parent)
            self.df = df
            self.setWindowTitle("Contour Plot")
            self.setMinimumSize(900, 700)
            self._setup_ui()

        def _setup_ui(self) -> None:
            from plot_engine.pyqt6_widget import PlotWidget

            layout = QVBoxLayout(self)

            config_group = QGroupBox("Configuration")
            config_layout = QFormLayout(config_group)

            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

            self._x_combo = QComboBox()
            self._x_combo.addItems(numeric_cols)
            config_layout.addRow("X Column:", self._x_combo)

            self._y_combo = QComboBox()
            self._y_combo.addItems(numeric_cols)
            if len(numeric_cols) > 1:
                self._y_combo.setCurrentIndex(1)
            config_layout.addRow("Y Column:", self._y_combo)

            self._z_combo = QComboBox()
            self._z_combo.addItems(numeric_cols)
            if len(numeric_cols) > 2:
                self._z_combo.setCurrentIndex(2)
            config_layout.addRow("Z Column:", self._z_combo)

            self._levels_spin = QSpinBox()
            self._levels_spin.setRange(5, 100)
            self._levels_spin.setValue(20)
            config_layout.addRow("Contour Levels:", self._levels_spin)

            self._filled_check = QCheckBox("Filled Contour")
            self._filled_check.setChecked(True)
            config_layout.addRow("", self._filled_check)

            self._labels_check = QCheckBox("Show Labels")
            config_layout.addRow("", self._labels_check)

            self._colormap_combo = QComboBox()
            self._colormap_combo.addItems(
                [
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "coolwarm",
                    "RdBu",
                    "YlGnBu",
                    "Spectral",
                    "jet",
                ]
            )
            config_layout.addRow("Colormap:", self._colormap_combo)

            self._resolution_spin = QSpinBox()
            self._resolution_spin.setRange(20, 500)
            self._resolution_spin.setValue(100)
            config_layout.addRow("Grid Resolution:", self._resolution_spin)

            layout.addWidget(config_group)

            plot_btn = QPushButton("Generate Contour Plot")
            plot_btn.clicked.connect(self._generate_plot)
            layout.addWidget(plot_btn)

            self._plot_widget = PlotWidget(self)
            layout.addWidget(self._plot_widget, stretch=1)

        def _generate_plot(self) -> None:
            from plot_engine.contour import scatter_to_grid
            from plot_engine.specs import AxisSpec, ContourPlotSpec

            x_col = self._x_combo.currentText()
            y_col = self._y_combo.currentText()
            z_col = self._z_combo.currentText()
            if not all([x_col, y_col, z_col]):
                return

            try:
                x = self.df[x_col].values.astype(float)
                y = self.df[y_col].values.astype(float)
                z = self.df[z_col].values.astype(float)

                x_grid, y_grid, z_grid = scatter_to_grid(
                    x,
                    y,
                    z,
                    resolution=self._resolution_spin.value(),
                )
                z_grid = np.nan_to_num(z_grid, nan=0.0)

                spec = ContourPlotSpec(
                    title=f"Contour: {z_col} vs ({x_col}, {y_col})",
                    z_data=z_grid.tolist(),
                    x_grid=x_grid.tolist(),
                    y_grid=y_grid.tolist(),
                    x_axis=AxisSpec(label=x_col),
                    y_axis=AxisSpec(label=y_col),
                    levels=self._levels_spin.value(),
                    filled=self._filled_check.isChecked(),
                    show_labels=self._labels_check.isChecked(),
                    colormap=self._colormap_combo.currentText(),
                )
                self._plot_widget.set_spec(spec)
            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.error(f"Contour plot failed: {e}")

    class HeatmapDialog(QDialog):
        """Dialog for creating heatmaps (correlation matrix or custom)."""

        def __init__(
            self,
            df: pd.DataFrame,
            parent: QWidget | None = None,
        ) -> None:
            assert df is not None, "df must be provided"
            super().__init__(parent)
            self.df = df
            self.setWindowTitle("Heatmap")
            self.setMinimumSize(800, 700)
            self._setup_ui()

        def _setup_ui(self) -> None:
            from plot_engine.pyqt6_widget import PlotWidget

            layout = QVBoxLayout(self)

            config_group = QGroupBox("Configuration")
            config_layout = QFormLayout(config_group)

            self._mode_combo = QComboBox()
            self._mode_combo.addItems(["Correlation Matrix", "Custom Z Data"])
            config_layout.addRow("Mode:", self._mode_combo)

            self._colormap_combo = QComboBox()
            self._colormap_combo.addItems(
                [
                    "YlGnBu",
                    "viridis",
                    "coolwarm",
                    "RdBu",
                    "Spectral",
                    "plasma",
                ]
            )
            config_layout.addRow("Colormap:", self._colormap_combo)

            self._annotate_check = QCheckBox("Show Values")
            self._annotate_check.setChecked(True)
            config_layout.addRow("", self._annotate_check)

            layout.addWidget(config_group)

            plot_btn = QPushButton("Generate Heatmap")
            plot_btn.clicked.connect(self._generate_plot)
            layout.addWidget(plot_btn)

            self._plot_widget = PlotWidget(self)
            layout.addWidget(self._plot_widget, stretch=1)

        def _generate_plot(self) -> None:
            from plot_engine.contour import correlation_matrix
            from plot_engine.specs import HeatmapSpec

            try:
                numeric_df = self.df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    return

                if self._mode_combo.currentText() == "Correlation Matrix":
                    corr_mat, labels = correlation_matrix(
                        numeric_df.values, list(numeric_df.columns)
                    )
                    spec = HeatmapSpec(
                        title="Correlation Matrix",
                        z_data=np.round(corr_mat, 3).tolist(),
                        x_labels=labels,
                        y_labels=labels,
                        colormap=self._colormap_combo.currentText(),
                        annotate=self._annotate_check.isChecked(),
                    )
                else:
                    cols = numeric_df.columns[: min(20, len(numeric_df.columns))]
                    data = numeric_df[cols].head(20).values
                    spec = HeatmapSpec(
                        title="Data Heatmap",
                        z_data=np.nan_to_num(data, nan=0.0).tolist(),
                        x_labels=list(cols),
                        y_labels=[str(i) for i in range(data.shape[0])],
                        colormap=self._colormap_combo.currentText(),
                        annotate=self._annotate_check.isChecked(),
                    )

                self._plot_widget.set_spec(spec)
            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
                logger.error(f"Heatmap generation failed: {e}")

    class FilterComparisonDialog(QDialog):
        """Dialog for comparing original vs filtered signals."""

        def __init__(
            self,
            original_df: pd.DataFrame,
            filtered_df: pd.DataFrame,
            time_col: str,
            signals: list[str],
            parent: QWidget | None = None,
        ) -> None:
            assert original_df is not None, "original_df must be provided"
            super().__init__(parent)
            self.original_df = original_df
            self.filtered_df = filtered_df
            self.time_col = time_col
            self.signals = signals
            self.setWindowTitle("Filter Comparison")
            self.setMinimumSize(1000, 700)
            self._setup_ui()
            self._generate_plot()

        def _setup_ui(self) -> None:
            from plot_engine.pyqt6_widget import PlotWidget

            layout = QVBoxLayout(self)

            config_layout = QHBoxLayout()
            self._diff_check = QCheckBox("Show Difference")
            self._diff_check.setChecked(True)
            self._diff_check.toggled.connect(self._generate_plot)
            config_layout.addWidget(self._diff_check)
            config_layout.addStretch()
            layout.addLayout(config_layout)

            self._plot_widget = PlotWidget(self)
            layout.addWidget(self._plot_widget, stretch=1)

        def _generate_plot(self) -> None:
            from plot_engine.specs import (
                AxisSpec,
                FilterComparisonSpec,
                SeriesData,
                SeriesStyle,
            )

            try:
                time_data = self.original_df[self.time_col].values.astype(float)
                orig_series = []
                filt_series = []

                for sig in self.signals:
                    if sig not in self.original_df.columns:
                        continue
                    orig_y = self.original_df[sig].values.astype(float)
                    orig_series.append(
                        SeriesData(
                            name=sig,
                            x=time_data.tolist(),
                            y=orig_y.tolist(),
                            style=SeriesStyle(line_style="solid"),
                        )
                    )

                    if sig in self.filtered_df.columns:
                        filt_y = self.filtered_df[sig].values.astype(float)
                        filt_series.append(
                            SeriesData(
                                name=sig,
                                x=time_data.tolist(),
                                y=filt_y.tolist(),
                                style=SeriesStyle(line_style="dashed"),
                            )
                        )

                spec = FilterComparisonSpec(
                    title="Original vs Filtered Signals",
                    x_axis=AxisSpec(label=self.time_col),
                    y_axis=AxisSpec(label="Value"),
                    original_series=orig_series,
                    filtered_series=filt_series,
                    show_difference=self._diff_check.isChecked(),
                )
                self._plot_widget.set_spec(spec)
            except (RuntimeError, AttributeError) as e:
                logger.error(f"Filter comparison failed: {e}")
