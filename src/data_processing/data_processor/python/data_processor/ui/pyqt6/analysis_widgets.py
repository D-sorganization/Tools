"""PyQt6 Widgets for Statistical Analysis Features.

Provides UI components for:
- PCA Analysis panel
- ANOVA Analysis panel
- Regression Analysis panel
- Surface Plot configuration
- Neural Network configuration
- Script generation interface
- Contour Plot dialog
- Heatmap dialog
- Filter Comparison dialog
- Chart Style panel

All widgets follow PyQt6 patterns and integrate with the main window.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    # Create dummy classes for type hints
    QWidget = object  # type: ignore[misc,assignment]

    def pyqtSignal(*args: object) -> None:  # type: ignore[no-redef]  # noqa: N802
        return None


logger = logging.getLogger(__name__)


if PYQT6_AVAILABLE:

    class VariableSelector(QWidget):
        """Widget for selecting variables from a dataset."""

        selection_changed = pyqtSignal(list)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            # Search box
            self.search_edit = QLineEdit()
            self.search_edit.setPlaceholderText("Search variables...")
            self.search_edit.textChanged.connect(self._filter_list)
            layout.addWidget(self.search_edit)

            # Variable list
            self.list_widget = QListWidget()
            self.list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
            layout.addWidget(self.list_widget)

            # Select all/none buttons
            btn_layout = QHBoxLayout()
            self.select_all_btn = QPushButton("Select All")
            self.select_all_btn.clicked.connect(self._select_all)
            self.select_none_btn = QPushButton("Select None")
            self.select_none_btn.clicked.connect(self._select_none)
            btn_layout.addWidget(self.select_all_btn)
            btn_layout.addWidget(self.select_none_btn)
            layout.addLayout(btn_layout)

        def set_variables(self, variables: list[str]) -> None:
            """Set the available variables."""
            self.list_widget.clear()
            for var in variables:
                item = QListWidgetItem(var)
                self.list_widget.addItem(item)

        def get_selected(self) -> list[str]:
            """Get currently selected variables."""
            return [item.text() for item in self.list_widget.selectedItems()]

        def _filter_list(self, text: str) -> None:
            """Filter list based on search text."""
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item is not None:
                    item.setHidden(text.lower() not in item.text().lower())

        def _select_all(self) -> None:
            """Select all visible items."""
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item is not None and not item.isHidden():
                    item.setSelected(True)

        def _select_none(self) -> None:
            """Deselect all items."""
            self.list_widget.clearSelection()

        def _on_selection_changed(self) -> None:
            """Emit signal when selection changes."""
            self.selection_changed.emit(self.get_selected())

    class PCAWidget(QWidget):
        """Widget for PCA analysis configuration and results."""

        analysis_requested = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            # Configuration group
            config_group = QGroupBox("Configuration")
            config_layout = QFormLayout(config_group)

            self.standardize_check = QCheckBox("Standardize Data")
            self.standardize_check.setChecked(True)
            config_layout.addRow("", self.standardize_check)

            self.n_components_spin = QSpinBox()
            self.n_components_spin.setRange(0, 100)
            self.n_components_spin.setSpecialValueText("All")
            config_layout.addRow("Components:", self.n_components_spin)

            self.variance_threshold_spin = QDoubleSpinBox()
            self.variance_threshold_spin.setRange(0.5, 1.0)
            self.variance_threshold_spin.setValue(0.95)
            self.variance_threshold_spin.setSingleStep(0.01)
            config_layout.addRow("Variance Threshold:", self.variance_threshold_spin)

            layout.addWidget(config_group)

            # Variable selector
            var_group = QGroupBox("Variables")
            var_layout = QVBoxLayout(var_group)
            self.variable_selector = VariableSelector()
            var_layout.addWidget(self.variable_selector)
            layout.addWidget(var_group)

            # Run button
            self.run_btn = QPushButton("Run PCA Analysis")
            self.run_btn.clicked.connect(self._run_analysis)
            layout.addWidget(self.run_btn)

            # Results area
            self.results_tabs = QTabWidget()

            # Summary tab
            self.summary_text = QTextEdit()
            self.summary_text.setReadOnly(True)
            self.results_tabs.addTab(self.summary_text, "Summary")

            # Components tab
            self.components_table = QTableWidget()
            self.results_tabs.addTab(self.components_table, "Components")

            # Loadings tab
            self.loadings_table = QTableWidget()
            self.results_tabs.addTab(self.loadings_table, "Loadings")

            layout.addWidget(self.results_tabs)

        def set_dataframe(self, df: pd.DataFrame) -> None:
            """Set the DataFrame and update variable list."""
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.variable_selector.set_variables(numeric_cols)

        def _run_analysis(self) -> None:
            """Emit signal to run analysis."""
            config = {
                "standardize": self.standardize_check.isChecked(),
                "n_components": self.n_components_spin.value() or None,
                "variance_threshold": self.variance_threshold_spin.value(),
                "variables": self.variable_selector.get_selected(),
            }
            self.analysis_requested.emit(config)

        def display_results(self, result: Any) -> None:
            """Display PCA results."""
            # Summary
            lines = [
                f"Number of components: {result.n_components}",
                f"Number of features: {result.n_features}",
                f"Number of samples: {result.n_samples}",
                f"Total variance explained: {result.total_variance_explained:.2%}",
                f"Kaiser criterion components: {result.kaiser_criterion_components}",
                f"Elbow point components: {result.elbow_point_components}",
                "",
                "Feature Importance:",
            ]
            for name, imp in sorted(
                result.feature_importance.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {name}: {imp:.4f}")

            self.summary_text.setText("\n".join(lines))

            # Components table
            self.components_table.setRowCount(len(result.components))
            self.components_table.setColumnCount(4)
            self.components_table.setHorizontalHeaderLabels(
                ["Component", "Variance", "% Variance", "Cumulative %"]
            )

            for i, comp in enumerate(result.components):
                self.components_table.setItem(i, 0, QTableWidgetItem(f"PC{comp.index}"))
                self.components_table.setItem(
                    i, 1, QTableWidgetItem(f"{comp.explained_variance:.4f}")
                )
                self.components_table.setItem(
                    i, 2, QTableWidgetItem(f"{comp.explained_variance_ratio:.2%}")
                )
                self.components_table.setItem(
                    i, 3, QTableWidgetItem(f"{comp.cumulative_variance_ratio:.2%}")
                )

            # Loadings table
            loading_df = result.loading_matrix
            self.loadings_table.setRowCount(len(loading_df))
            self.loadings_table.setColumnCount(len(loading_df.columns))
            self.loadings_table.setHorizontalHeaderLabels(list(loading_df.columns))
            self.loadings_table.setVerticalHeaderLabels(list(loading_df.index))

            for i, row in enumerate(loading_df.index):
                for j, col in enumerate(loading_df.columns):
                    value = loading_df.loc[row, col]
                    self.loadings_table.setItem(i, j, QTableWidgetItem(f"{value:.4f}"))

    class ANOVAWidget(QWidget):
        """Widget for ANOVA analysis configuration and results."""

        analysis_requested = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            # Type selection
            type_group = QGroupBox("Analysis Type")
            type_layout = QVBoxLayout(type_group)

            self.type_combo = QComboBox()
            self.type_combo.addItems(
                ["One-Way ANOVA", "Two-Way ANOVA", "Repeated Measures"]
            )
            self.type_combo.currentIndexChanged.connect(self._update_config_ui)
            type_layout.addWidget(self.type_combo)

            layout.addWidget(type_group)

            # Configuration (stacked widget for different types)
            self.config_stack = QStackedWidget()

            # One-way config
            oneway_widget = QWidget()
            oneway_layout = QFormLayout(oneway_widget)
            self.dependent_combo = QComboBox()
            oneway_layout.addRow("Dependent Variable:", self.dependent_combo)
            self.group_combo = QComboBox()
            oneway_layout.addRow("Grouping Variable:", self.group_combo)
            self.config_stack.addWidget(oneway_widget)

            # Two-way config
            twoway_widget = QWidget()
            twoway_layout = QFormLayout(twoway_widget)
            self.dependent_combo_2 = QComboBox()
            twoway_layout.addRow("Dependent Variable:", self.dependent_combo_2)
            self.factor_a_combo = QComboBox()
            twoway_layout.addRow("Factor A:", self.factor_a_combo)
            self.factor_b_combo = QComboBox()
            twoway_layout.addRow("Factor B:", self.factor_b_combo)
            self.interaction_check = QCheckBox("Test Interaction")
            self.interaction_check.setChecked(True)
            twoway_layout.addRow("", self.interaction_check)
            self.config_stack.addWidget(twoway_widget)

            # Repeated measures config
            rm_widget = QWidget()
            rm_layout = QFormLayout(rm_widget)
            self.subject_combo = QComboBox()
            rm_layout.addRow("Subject ID:", self.subject_combo)
            self.measures_list = QListWidget()
            self.measures_list.setSelectionMode(
                QListWidget.SelectionMode.MultiSelection
            )
            rm_layout.addRow("Measures:", self.measures_list)
            self.config_stack.addWidget(rm_widget)

            layout.addWidget(self.config_stack)

            # Post-hoc options
            posthoc_group = QGroupBox("Post-hoc Tests")
            posthoc_layout = QFormLayout(posthoc_group)

            self.posthoc_combo = QComboBox()
            self.posthoc_combo.addItems(["Tukey HSD", "Bonferroni", "Scheffé", "None"])
            posthoc_layout.addRow("Method:", self.posthoc_combo)

            self.alpha_spin = QDoubleSpinBox()
            self.alpha_spin.setRange(0.001, 0.1)
            self.alpha_spin.setValue(0.05)
            self.alpha_spin.setSingleStep(0.01)
            posthoc_layout.addRow("Alpha:", self.alpha_spin)

            layout.addWidget(posthoc_group)

            # Run button
            self.run_btn = QPushButton("Run ANOVA")
            self.run_btn.clicked.connect(self._run_analysis)
            layout.addWidget(self.run_btn)

            # Results
            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            layout.addWidget(self.results_text)

        def _update_config_ui(self, index: int) -> None:
            """Update config UI based on selected ANOVA type."""
            self.config_stack.setCurrentIndex(index)

        def set_dataframe(self, df: pd.DataFrame) -> None:
            """Set DataFrame and update variable combos."""
            all_cols = list(df.columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            for combo in [self.dependent_combo, self.dependent_combo_2]:
                combo.clear()
                combo.addItems(numeric_cols)

            for combo in [
                self.group_combo,
                self.factor_a_combo,
                self.factor_b_combo,
                self.subject_combo,
            ]:
                combo.clear()
                combo.addItems(all_cols)

            self.measures_list.clear()
            for col in numeric_cols:
                self.measures_list.addItem(col)

        def _run_analysis(self) -> None:
            """Emit signal to run analysis."""
            config = {
                "type": self.type_combo.currentText(),
                "alpha": self.alpha_spin.value(),
                "posthoc": self.posthoc_combo.currentText(),
            }

            idx = self.type_combo.currentIndex()
            if idx == 0:  # One-way
                config["dependent"] = self.dependent_combo.currentText()
                config["group"] = self.group_combo.currentText()
            elif idx == 1:  # Two-way
                config["dependent"] = self.dependent_combo_2.currentText()
                config["factor_a"] = self.factor_a_combo.currentText()
                config["factor_b"] = self.factor_b_combo.currentText()
                config["interaction"] = self.interaction_check.isChecked()
            else:  # Repeated measures
                config["subject"] = self.subject_combo.currentText()
                config["measures"] = [
                    item.text() for item in self.measures_list.selectedItems()
                ]

            self.analysis_requested.emit(config)

        def display_results(self, report: str) -> None:
            """Display ANOVA results."""
            self.results_text.setText(report)

    class RegressionWidget(QWidget):
        """Widget for regression analysis."""

        analysis_requested = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            # Target variable
            target_group = QGroupBox("Target Variable")
            target_layout = QFormLayout(target_group)
            self.target_combo = QComboBox()
            target_layout.addRow("Target:", self.target_combo)
            layout.addWidget(target_group)

            # Predictors
            pred_group = QGroupBox("Predictors")
            pred_layout = QVBoxLayout(pred_group)
            self.predictor_selector = VariableSelector()
            pred_layout.addWidget(self.predictor_selector)
            layout.addWidget(pred_group)

            # Model options
            options_group = QGroupBox("Model Options")
            options_layout = QFormLayout(options_group)

            self.regularization_combo = QComboBox()
            self.regularization_combo.addItems(
                ["None", "Ridge", "Lasso", "Elastic Net"]
            )
            options_layout.addRow("Regularization:", self.regularization_combo)

            self.alpha_spin = QDoubleSpinBox()
            self.alpha_spin.setRange(0, 100)
            self.alpha_spin.setValue(1.0)
            options_layout.addRow("Alpha:", self.alpha_spin)

            self.polynomial_spin = QSpinBox()
            self.polynomial_spin.setRange(1, 5)
            self.polynomial_spin.setValue(1)
            options_layout.addRow("Polynomial Degree:", self.polynomial_spin)

            self.interactions_check = QCheckBox("Include Interactions")
            options_layout.addRow("", self.interactions_check)

            self.selection_combo = QComboBox()
            self.selection_combo.addItems(["None", "Forward", "Backward", "Stepwise"])
            options_layout.addRow("Feature Selection:", self.selection_combo)

            layout.addWidget(options_group)

            # Run button
            self.run_btn = QPushButton("Run Regression")
            self.run_btn.clicked.connect(self._run_analysis)
            layout.addWidget(self.run_btn)

            # Results tabs
            self.results_tabs = QTabWidget()

            self.summary_text = QTextEdit()
            self.summary_text.setReadOnly(True)
            self.results_tabs.addTab(self.summary_text, "Summary")

            self.coefficients_table = QTableWidget()
            self.results_tabs.addTab(self.coefficients_table, "Coefficients")

            self.diagnostics_text = QTextEdit()
            self.diagnostics_text.setReadOnly(True)
            self.results_tabs.addTab(self.diagnostics_text, "Diagnostics")

            layout.addWidget(self.results_tabs)

        def set_dataframe(self, df: pd.DataFrame) -> None:
            """Set DataFrame and update variable lists."""
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.target_combo.clear()
            self.target_combo.addItems(numeric_cols)
            self.predictor_selector.set_variables(numeric_cols)

        def _run_analysis(self) -> None:
            """Emit signal to run analysis."""
            config = {
                "target": self.target_combo.currentText(),
                "predictors": self.predictor_selector.get_selected(),
                "regularization": self.regularization_combo.currentText().lower(),
                "alpha": self.alpha_spin.value(),
                "polynomial_degree": self.polynomial_spin.value(),
                "interactions": self.interactions_check.isChecked(),
                "selection": self.selection_combo.currentText().lower(),
            }
            self.analysis_requested.emit(config)

        def display_results(self, result: Any, report: str) -> None:
            """Display regression results."""
            self.summary_text.setText(report)

            # Coefficients table
            n_coefs = len(result.coefficients)
            self.coefficients_table.setRowCount(n_coefs + 1)
            self.coefficients_table.setColumnCount(6)
            self.coefficients_table.setHorizontalHeaderLabels(
                ["Variable", "Estimate", "Std.Error", "t-stat", "p-value", "VIF"]
            )

            # Intercept row
            self.coefficients_table.setItem(0, 0, QTableWidgetItem("(Intercept)"))
            self.coefficients_table.setItem(
                0, 1, QTableWidgetItem(f"{result.intercept:.4f}")
            )

            for i, coef in enumerate(result.coefficients):
                row = i + 1
                self.coefficients_table.setItem(row, 0, QTableWidgetItem(coef.name))
                self.coefficients_table.setItem(
                    row, 1, QTableWidgetItem(f"{coef.estimate:.4f}")
                )
                self.coefficients_table.setItem(
                    row, 2, QTableWidgetItem(f"{coef.std_error:.4f}")
                )
                self.coefficients_table.setItem(
                    row, 3, QTableWidgetItem(f"{coef.t_statistic:.4f}")
                )
                self.coefficients_table.setItem(
                    row, 4, QTableWidgetItem(f"{coef.p_value:.4e}")
                )
                self.coefficients_table.setItem(
                    row, 5, QTableWidgetItem(f"{coef.vif:.2f}")
                )

    class SurfacePlotWidget(QWidget):
        """Widget for 3D surface plot configuration."""

        plot_requested = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            # Axis selection
            axis_group = QGroupBox("Axis Selection")
            axis_layout = QFormLayout(axis_group)

            self.x_combo = QComboBox()
            axis_layout.addRow("X Axis:", self.x_combo)

            self.y_combo = QComboBox()
            axis_layout.addRow("Y Axis:", self.y_combo)

            self.z_combo = QComboBox()
            axis_layout.addRow("Z Axis:", self.z_combo)

            layout.addWidget(axis_group)

            # Grid settings
            grid_group = QGroupBox("Grid Settings")
            grid_layout = QFormLayout(grid_group)

            self.resolution_spin = QSpinBox()
            self.resolution_spin.setRange(10, 200)
            self.resolution_spin.setValue(50)
            grid_layout.addRow("Resolution:", self.resolution_spin)

            layout.addWidget(grid_group)

            # Interpolation
            interp_group = QGroupBox("Interpolation")
            interp_layout = QFormLayout(interp_group)

            self.interp_combo = QComboBox()
            self.interp_combo.addItems(
                [
                    "Linear",
                    "Cubic",
                    "Nearest",
                    "RBF Thin Plate",
                    "RBF Multiquadric",
                    "RBF Gaussian",
                ]
            )
            interp_layout.addRow("Method:", self.interp_combo)

            layout.addWidget(interp_group)

            # Smoothing
            smooth_group = QGroupBox("Smoothing")
            smooth_layout = QFormLayout(smooth_group)

            self.smooth_combo = QComboBox()
            self.smooth_combo.addItems(
                ["None", "Gaussian", "Median", "Uniform", "Savitzky-Golay"]
            )
            smooth_layout.addRow("Method:", self.smooth_combo)

            self.sigma_spin = QDoubleSpinBox()
            self.sigma_spin.setRange(0.1, 10)
            self.sigma_spin.setValue(1.0)
            smooth_layout.addRow("Sigma:", self.sigma_spin)

            self.kernel_spin = QSpinBox()
            self.kernel_spin.setRange(3, 21)
            self.kernel_spin.setValue(3)
            self.kernel_spin.setSingleStep(2)
            smooth_layout.addRow("Kernel Size:", self.kernel_spin)

            layout.addWidget(smooth_group)

            # Outlier removal
            outlier_group = QGroupBox("Outlier Handling")
            outlier_layout = QFormLayout(outlier_group)

            self.remove_outliers_check = QCheckBox("Remove Outliers")
            outlier_layout.addRow("", self.remove_outliers_check)

            self.threshold_spin = QDoubleSpinBox()
            self.threshold_spin.setRange(1, 10)
            self.threshold_spin.setValue(3.0)
            outlier_layout.addRow("Z-Score Threshold:", self.threshold_spin)

            layout.addWidget(outlier_group)

            # Appearance
            appear_group = QGroupBox("Appearance")
            appear_layout = QFormLayout(appear_group)

            self.colormap_combo = QComboBox()
            self.colormap_combo.addItems(
                [
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "coolwarm",
                    "RdBu",
                    "jet",
                    "terrain",
                ]
            )
            appear_layout.addRow("Colormap:", self.colormap_combo)

            self.alpha_spin = QDoubleSpinBox()
            self.alpha_spin.setRange(0.1, 1.0)
            self.alpha_spin.setValue(0.8)
            appear_layout.addRow("Alpha:", self.alpha_spin)

            self.show_scatter_check = QCheckBox("Show Data Points")
            self.show_scatter_check.setChecked(True)
            appear_layout.addRow("", self.show_scatter_check)

            layout.addWidget(appear_group)

            # Plot button
            self.plot_btn = QPushButton("Create Surface Plot")
            self.plot_btn.clicked.connect(self._create_plot)
            layout.addWidget(self.plot_btn)

        def set_dataframe(self, df: pd.DataFrame) -> None:
            """Set DataFrame and update variable combos."""
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for combo in [self.x_combo, self.y_combo, self.z_combo]:
                combo.clear()
                combo.addItems(numeric_cols)

        def _create_plot(self) -> None:
            """Emit signal to create surface plot."""
            config = {
                "x_column": self.x_combo.currentText(),
                "y_column": self.y_combo.currentText(),
                "z_column": self.z_combo.currentText(),
                "grid_resolution": self.resolution_spin.value(),
                "interpolation": self.interp_combo.currentText()
                .lower()
                .replace(" ", "_"),
                "smoothing": self.smooth_combo.currentText().lower(),
                "smoothing_sigma": self.sigma_spin.value(),
                "smoothing_kernel": self.kernel_spin.value(),
                "remove_outliers": self.remove_outliers_check.isChecked(),
                "outlier_threshold": self.threshold_spin.value(),
                "colormap": self.colormap_combo.currentText(),
                "alpha": self.alpha_spin.value(),
                "show_scatter": self.show_scatter_check.isChecked(),
            }
            self.plot_requested.emit(config)

    class NeuralNetworkWidget(QWidget):
        """Widget for neural network configuration and training."""

        train_requested = pyqtSignal(dict)
        export_requested = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            # Tabs for different sections
            tabs = QTabWidget()

            # Architecture tab
            arch_widget = QWidget()
            arch_layout = QVBoxLayout(arch_widget)

            arch_group = QGroupBox("Network Architecture")
            arch_form = QFormLayout(arch_group)

            self.network_type_combo = QComboBox()
            self.network_type_combo.addItems(["MLP", "LSTM", "GRU", "CNN 1D"])
            arch_form.addRow("Network Type:", self.network_type_combo)

            self.hidden_layers_edit = QLineEdit("128, 64, 32")
            arch_form.addRow("Hidden Layers:", self.hidden_layers_edit)

            self.activation_combo = QComboBox()
            self.activation_combo.addItems(
                ["ReLU", "Leaky ReLU", "ELU", "SELU", "Tanh", "Sigmoid", "GELU"]
            )
            arch_form.addRow("Activation:", self.activation_combo)

            self.dropout_spin = QDoubleSpinBox()
            self.dropout_spin.setRange(0, 0.9)
            self.dropout_spin.setValue(0.2)
            arch_form.addRow("Dropout:", self.dropout_spin)

            arch_layout.addWidget(arch_group)
            tabs.addTab(arch_widget, "Architecture")

            # Training tab
            train_widget = QWidget()
            train_layout = QVBoxLayout(train_widget)

            train_group = QGroupBox("Training Settings")
            train_form = QFormLayout(train_group)

            self.optimizer_combo = QComboBox()
            self.optimizer_combo.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
            train_form.addRow("Optimizer:", self.optimizer_combo)

            self.lr_spin = QDoubleSpinBox()
            self.lr_spin.setDecimals(5)
            self.lr_spin.setRange(0.00001, 1.0)
            self.lr_spin.setValue(0.001)
            train_form.addRow("Learning Rate:", self.lr_spin)

            self.batch_spin = QSpinBox()
            self.batch_spin.setRange(1, 1024)
            self.batch_spin.setValue(32)
            train_form.addRow("Batch Size:", self.batch_spin)

            self.epochs_spin = QSpinBox()
            self.epochs_spin.setRange(1, 10000)
            self.epochs_spin.setValue(100)
            train_form.addRow("Epochs:", self.epochs_spin)

            self.early_stop_spin = QSpinBox()
            self.early_stop_spin.setRange(1, 100)
            self.early_stop_spin.setValue(10)
            train_form.addRow("Early Stopping Patience:", self.early_stop_spin)

            train_layout.addWidget(train_group)
            tabs.addTab(train_widget, "Training")

            # Data tab
            data_widget = QWidget()
            data_layout = QVBoxLayout(data_widget)

            data_group = QGroupBox("Data Configuration")
            data_form = QFormLayout(data_group)

            self.target_list = QListWidget()
            self.target_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            data_form.addRow("Target(s):", self.target_list)

            self.feature_selector = VariableSelector()
            data_form.addRow("Features:", self.feature_selector)

            self.normalize_check = QCheckBox("Normalize Inputs")
            self.normalize_check.setChecked(True)
            data_form.addRow("", self.normalize_check)

            self.val_split_spin = QDoubleSpinBox()
            self.val_split_spin.setRange(0.05, 0.5)
            self.val_split_spin.setValue(0.2)
            data_form.addRow("Validation Split:", self.val_split_spin)

            data_layout.addWidget(data_group)
            tabs.addTab(data_widget, "Data")

            layout.addWidget(tabs)

            # Action buttons
            btn_layout = QHBoxLayout()

            self.train_btn = QPushButton("Train Model")
            self.train_btn.clicked.connect(self._request_train)
            btn_layout.addWidget(self.train_btn)

            self.export_pytorch_btn = QPushButton("Export PyTorch")
            self.export_pytorch_btn.clicked.connect(
                lambda: self._request_export("pytorch")
            )
            btn_layout.addWidget(self.export_pytorch_btn)

            self.export_tf_btn = QPushButton("Export TensorFlow")
            self.export_tf_btn.clicked.connect(
                lambda: self._request_export("tensorflow")
            )
            btn_layout.addWidget(self.export_tf_btn)

            layout.addLayout(btn_layout)

            # Progress and results
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            layout.addWidget(self.progress_bar)

            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            layout.addWidget(self.results_text)

        def set_dataframe(self, df: pd.DataFrame) -> None:
            """Set DataFrame and update variable lists."""
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            self.target_list.clear()
            for col in numeric_cols:
                self.target_list.addItem(col)

            self.feature_selector.set_variables(numeric_cols)

        def _get_config(self) -> dict[str, Any]:
            """Get current configuration."""
            hidden_layers = [
                int(x.strip())
                for x in self.hidden_layers_edit.text().split(",")
                if x.strip().isdigit()
            ]

            return {
                "network_type": self.network_type_combo.currentText().lower(),
                "hidden_layers": hidden_layers,
                "activation": self.activation_combo.currentText()
                .lower()
                .replace(" ", "_"),
                "dropout": self.dropout_spin.value(),
                "optimizer": self.optimizer_combo.currentText().lower(),
                "learning_rate": self.lr_spin.value(),
                "batch_size": self.batch_spin.value(),
                "epochs": self.epochs_spin.value(),
                "early_stopping": self.early_stop_spin.value(),
                "targets": [item.text() for item in self.target_list.selectedItems()],
                "features": self.feature_selector.get_selected(),
                "normalize": self.normalize_check.isChecked(),
                "validation_split": self.val_split_spin.value(),
            }

        def _request_train(self) -> None:
            """Emit signal to train model."""
            self.train_requested.emit(self._get_config())

        def _request_export(self, framework: str) -> None:
            """Emit signal to export script."""
            config = self._get_config()
            config["framework"] = framework
            self.export_requested.emit(config)

        def display_results(self, result: Any) -> None:
            """Display training results."""
            lines = [
                "Training Complete",
                "=" * 40,
                f"Best Epoch: {result.best_epoch}",
                f"Best Validation Loss: {result.best_val_loss:.6f}",
                f"Final Training Loss: {result.final_train_loss:.6f}",
                f"Training Time: {result.training_time_seconds:.2f}s",
            ]
            if result.test_loss is not None:
                lines.append(f"Test Loss: {result.test_loss:.6f}")
            if result.stopped_early:
                lines.append("(Stopped early)")

            self.results_text.setText("\n".join(lines))

    class ScriptGeneratorWidget(QWidget):
        """Widget for script generation."""

        generate_requested = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)

            # Pipeline steps
            steps_group = QGroupBox("Pipeline Steps")
            steps_layout = QVBoxLayout(steps_group)

            self.steps_list = QListWidget()
            steps_layout.addWidget(self.steps_list)

            # Step buttons
            step_btn_layout = QHBoxLayout()
            self.add_step_btn = QPushButton("Add Step")
            self.remove_step_btn = QPushButton("Remove Step")
            self.move_up_btn = QPushButton("↑")
            self.move_down_btn = QPushButton("↓")
            step_btn_layout.addWidget(self.add_step_btn)
            step_btn_layout.addWidget(self.remove_step_btn)
            step_btn_layout.addWidget(self.move_up_btn)
            step_btn_layout.addWidget(self.move_down_btn)
            steps_layout.addLayout(step_btn_layout)

            layout.addWidget(steps_group)

            # Output options
            output_group = QGroupBox("Output Options")
            output_layout = QFormLayout(output_group)

            self.include_imports_check = QCheckBox("Include Imports")
            self.include_imports_check.setChecked(True)
            output_layout.addRow("", self.include_imports_check)

            self.include_logging_check = QCheckBox("Include Logging")
            self.include_logging_check.setChecked(True)
            output_layout.addRow("", self.include_logging_check)

            self.use_argparse_check = QCheckBox("Use Argparse")
            self.use_argparse_check.setChecked(True)
            output_layout.addRow("", self.use_argparse_check)

            layout.addWidget(output_group)

            # Generate buttons
            btn_layout = QHBoxLayout()

            self.generate_btn = QPushButton("Generate Script")
            self.generate_btn.clicked.connect(self._generate_script)
            btn_layout.addWidget(self.generate_btn)

            self.save_btn = QPushButton("Save Script")
            self.save_btn.clicked.connect(self._save_script)
            btn_layout.addWidget(self.save_btn)

            layout.addLayout(btn_layout)

            # Script preview
            self.script_preview = QPlainTextEdit()
            self.script_preview.setReadOnly(True)
            layout.addWidget(self.script_preview)

        def _generate_script(self) -> None:
            """Emit signal to generate script."""
            config = {
                "include_imports": self.include_imports_check.isChecked(),
                "include_logging": self.include_logging_check.isChecked(),
                "use_argparse": self.use_argparse_check.isChecked(),
            }
            self.generate_requested.emit(config)

        def _save_script(self) -> None:
            """Save generated script to file."""
            text = self.script_preview.toPlainText()
            if not text:
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Script", "", "Python Files (*.py)"
            )
            if file_path:
                Path(file_path).write_text(text)

        def set_script(self, script: str) -> None:
            """Set the script preview text."""
            self.script_preview.setPlainText(script)

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

    class ContourPlotDialog(QDialog):
        """Dialog for creating contour plots from DataFrame columns."""

        def __init__(
            self,
            df: pd.DataFrame,
            parent: QWidget | None = None,
        ) -> None:
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
