"""PyQt6 Widgets for Visualization and Neural Network.

Provides UI components for:
- 3D Surface Plot configuration
- Neural Network configuration and training
- Script generation interface
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
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLineEdit,
        QListWidget,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
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
    from .statistical_widgets import VariableSelector

    class SurfacePlotWidget(QWidget):
        """Widget for 3D surface plot configuration."""

        plot_requested = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._setup_ui()

        def _setup_ui(self) -> None:
            layout = QVBoxLayout(self)
            layout.addWidget(self._create_axis_group())
            layout.addWidget(self._create_grid_group())
            layout.addWidget(self._create_interpolation_group())
            layout.addWidget(self._create_smoothing_group())
            layout.addWidget(self._create_outlier_group())
            layout.addWidget(self._create_appearance_group())

            self.plot_btn = QPushButton("Create Surface Plot")
            self.plot_btn.clicked.connect(self._create_plot)
            layout.addWidget(self.plot_btn)

        def _create_axis_group(self) -> QGroupBox:
            """Create axis selection group."""
            axis_group = QGroupBox("Axis Selection")
            axis_layout = QFormLayout(axis_group)
            self.x_combo = QComboBox()
            axis_layout.addRow("X Axis:", self.x_combo)
            self.y_combo = QComboBox()
            axis_layout.addRow("Y Axis:", self.y_combo)
            self.z_combo = QComboBox()
            axis_layout.addRow("Z Axis:", self.z_combo)
            return axis_group

        def _create_grid_group(self) -> QGroupBox:
            """Create grid settings group."""
            grid_group = QGroupBox("Grid Settings")
            grid_layout = QFormLayout(grid_group)
            self.resolution_spin = QSpinBox()
            self.resolution_spin.setRange(10, 200)
            self.resolution_spin.setValue(50)
            grid_layout.addRow("Resolution:", self.resolution_spin)
            return grid_group

        def _create_interpolation_group(self) -> QGroupBox:
            """Create interpolation settings group."""
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
            return interp_group

        def _create_smoothing_group(self) -> QGroupBox:
            """Create smoothing settings group."""
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
            return smooth_group

        def _create_outlier_group(self) -> QGroupBox:
            """Create outlier handling group."""
            outlier_group = QGroupBox("Outlier Handling")
            outlier_layout = QFormLayout(outlier_group)
            self.remove_outliers_check = QCheckBox("Remove Outliers")
            outlier_layout.addRow("", self.remove_outliers_check)
            self.threshold_spin = QDoubleSpinBox()
            self.threshold_spin.setRange(1, 10)
            self.threshold_spin.setValue(3.0)
            outlier_layout.addRow("Z-Score Threshold:", self.threshold_spin)
            return outlier_group

        def _create_appearance_group(self) -> QGroupBox:
            """Create appearance settings group."""
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
            return appear_group

        def set_dataframe(self, df: pd.DataFrame) -> None:
            """Set DataFrame and update variable combos."""
            if not (df is not None):
                raise ValueError("df must be provided")
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

            tabs = QTabWidget()
            tabs.addTab(self._create_architecture_tab(), "Architecture")
            tabs.addTab(self._create_training_tab(), "Training")
            tabs.addTab(self._create_data_tab(), "Data")
            layout.addWidget(tabs)

            layout.addLayout(self._create_action_buttons())

            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            layout.addWidget(self.progress_bar)

            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            layout.addWidget(self.results_text)

        def _create_architecture_tab(self) -> QWidget:
            """Create the network architecture configuration tab."""
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
            return arch_widget

        def _create_training_tab(self) -> QWidget:
            """Create the training settings tab."""
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
            return train_widget

        def _create_data_tab(self) -> QWidget:
            """Create the data configuration tab."""
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
            return data_widget

        def _create_action_buttons(self) -> QHBoxLayout:
            """Create the action button row."""
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

            return btn_layout

        def set_dataframe(self, df: pd.DataFrame) -> None:
            """Set DataFrame and update variable lists."""
            if not (df is not None):
                raise ValueError("df must be provided")
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
            if not (framework is not None):
                raise ValueError("framework must be provided")
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
