"""PyQt6 Widgets for Statistical Analysis Features.

This module is a backward-compatible facade that re-exports all analysis
widget classes from the decomposed submodules.

Submodules:
- statistical_widgets: VariableSelector, PCAWidget, ANOVAWidget, RegressionWidget
- visualization_widgets: SurfacePlotWidget, NeuralNetworkWidget, ScriptGeneratorWidget
- plot_dialogs: ContourPlotDialog, HeatmapDialog, FilterComparisonDialog
- chart_style_panel: ChartStylePanel
"""

from __future__ import annotations

import logging

import pandas as pd

try:
    from PyQt6.QtWidgets import (
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


# Re-export all widget classes for backward compatibility
# Only import when PyQt6 is available, as submodules define classes
# inside PYQT6_AVAILABLE guards
if PYQT6_AVAILABLE:
    from .chart_style_panel import ChartStylePanel  # noqa: E402
    from .plot_dialogs import (  # noqa: E402
        ContourPlotDialog,
        FilterComparisonDialog,
        HeatmapDialog,
    )
    from .statistical_widgets import (  # noqa: E402
        ANOVAWidget,
        PCAWidget,
        RegressionWidget,
        VariableSelector,
    )
    from .visualization_widgets import (  # noqa: E402
        NeuralNetworkWidget,
        ScriptGeneratorWidget,
        SurfacePlotWidget,
    )

logger = logging.getLogger(__name__)


if PYQT6_AVAILABLE:

    class AnalysisPanel(QWidget):
        """Main panel containing all analysis widgets."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            self.tabs = QTabWidget()

            # Add analysis widgets
            self.pca_widget = PCAWidget()
            self.tabs.addTab(self.pca_widget, "PCA")

            self.anova_widget = ANOVAWidget()
            self.tabs.addTab(self.anova_widget, "ANOVA")

            self.regression_widget = RegressionWidget()
            self.tabs.addTab(self.regression_widget, "Regression")

            self.surface_widget = SurfacePlotWidget()
            self.tabs.addTab(self.surface_widget, "Surface Plot")

            self.nn_widget = NeuralNetworkWidget()
            self.tabs.addTab(self.nn_widget, "Neural Network")

            self.script_widget = ScriptGeneratorWidget()
            self.tabs.addTab(self.script_widget, "Script Generator")

            layout.addWidget(self.tabs)

        def set_dataframe(self, df: pd.DataFrame) -> None:
            """Update all widgets with new DataFrame."""
            self.pca_widget.set_dataframe(df)
            self.anova_widget.set_dataframe(df)
            self.regression_widget.set_dataframe(df)
            self.surface_widget.set_dataframe(df)
            self.nn_widget.set_dataframe(df)


__all__ = [
    "VariableSelector",
    "PCAWidget",
    "ANOVAWidget",
    "RegressionWidget",
    "SurfacePlotWidget",
    "NeuralNetworkWidget",
    "ScriptGeneratorWidget",
    "AnalysisPanel",
    "ContourPlotDialog",
    "HeatmapDialog",
    "FilterComparisonDialog",
    "ChartStylePanel",
    "PYQT6_AVAILABLE",
]
