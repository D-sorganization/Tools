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
    from PyQt6.QtCore import pyqtSignal
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
        """Main panel containing all analysis widgets.

        The panel re-exposes aggregate request signals that forward from its
        internal child widgets. Consumers (e.g. ``MainWindow``) connect to
        these panel-level signals instead of reaching through the panel into
        its private child widgets, so the panel's internal composition can
        change without breaking its consumers.
        """

        # Aggregate signals forwarded from the internal child widgets. Each
        # carries the same ``dict`` payload as the originating child signal.
        pca_requested = pyqtSignal(dict)
        anova_requested = pyqtSignal(dict)
        regression_requested = pyqtSignal(dict)
        surface_requested = pyqtSignal(dict)
        nn_train_requested = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()
            self._connect_child_signals()

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

        def _connect_child_signals(self) -> None:
            """Forward each child widget's request signal to a panel signal."""
            self.pca_widget.analysis_requested.connect(self.pca_requested)
            self.anova_widget.analysis_requested.connect(self.anova_requested)
            self.regression_widget.analysis_requested.connect(self.regression_requested)
            self.surface_widget.plot_requested.connect(self.surface_requested)
            self.nn_widget.train_requested.connect(self.nn_train_requested)

        def set_dataframe(self, df: pd.DataFrame) -> None:
            """Update all widgets with new DataFrame."""
            if df is None:
                raise ValueError("df must be provided")
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
