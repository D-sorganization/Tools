"""PyQt6 Widgets for Statistical Analysis.

Provides UI components for:
- Variable selection (reusable)
- PCA Analysis panel
- ANOVA Analysis panel
- Regression Analysis panel
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
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
            if not (variables is not None):
                raise ValueError("variables must be provided")
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
            if not (df is not None):
                raise ValueError("df must be provided")
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
            lines.extend(
                [
                    f"  {name}: {imp:.4f}"
                    for (name, imp) in sorted(
                        result.feature_importance.items(), key=lambda x: x[1], reverse=True
                    )
                ]
            )

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
            self.type_combo.addItems(["One-Way ANOVA", "Two-Way ANOVA", "Repeated Measures"])
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
            self.measures_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
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
            if not (df is not None):
                raise ValueError("df must be provided")
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
                config["measures"] = [item.text() for item in self.measures_list.selectedItems()]

            self.analysis_requested.emit(config)

        def display_results(self, report: str) -> None:
            """Display ANOVA results."""
            if not (report is not None):
                raise ValueError("report must be provided")
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
            self.regularization_combo.addItems(["None", "Ridge", "Lasso", "Elastic Net"])
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
            if not (df is not None):
                raise ValueError("df must be provided")
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
            if not (report is not None):
                raise ValueError("report must be provided")
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
            self.coefficients_table.setItem(0, 1, QTableWidgetItem(f"{result.intercept:.4f}"))

            for i, coef in enumerate(result.coefficients):
                row = i + 1
                self.coefficients_table.setItem(row, 0, QTableWidgetItem(coef.name))
                self.coefficients_table.setItem(row, 1, QTableWidgetItem(f"{coef.estimate:.4f}"))
                self.coefficients_table.setItem(row, 2, QTableWidgetItem(f"{coef.std_error:.4f}"))
                self.coefficients_table.setItem(row, 3, QTableWidgetItem(f"{coef.t_statistic:.4f}"))
                self.coefficients_table.setItem(row, 4, QTableWidgetItem(f"{coef.p_value:.4e}"))
                self.coefficients_table.setItem(row, 5, QTableWidgetItem(f"{coef.vif:.2f}"))
