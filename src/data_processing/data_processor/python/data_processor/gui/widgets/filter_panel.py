"""Filter configuration panel widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

# Filter types and their parameters
FILTER_CONFIGS = {
    "Moving Average": {"ma_window": ("Window Size", "int", 10, 3, 1000)},
    "Butterworth Low-pass": {
        "bw_order": ("Order", "int", 3, 1, 10),
        "bw_cutoff": ("Cutoff", "float", 0.1, 0.01, 0.99),
    },
    "Butterworth High-pass": {
        "bw_order": ("Order", "int", 3, 1, 10),
        "bw_cutoff": ("Cutoff", "float", 0.1, 0.01, 0.99),
    },
    "Median Filter": {"median_kernel": ("Kernel Size", "int", 5, 3, 101)},
    "Hampel Filter": {
        "hampel_window": ("Window Size", "int", 5, 3, 100),
        "hampel_threshold": ("Threshold", "float", 3.0, 1.0, 10.0),
    },
    "Z-Score Filter": {"zscore_threshold": ("Threshold", "float", 3.0, 1.0, 10.0)},
    "Savitzky-Golay": {
        "savgol_window": ("Window Size", "int", 5, 3, 101),
        "savgol_polyorder": ("Poly Order", "int", 2, 1, 6),
    },
    "Gaussian Filter": {"gaussian_sigma": ("Sigma", "float", 1.0, 0.1, 100.0)},
}


class FilterPanel(QWidget):
    """Panel for configuring and applying filters."""

    # Signals
    filter_requested = pyqtSignal(dict)  # Filter configuration dict

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the filter panel."""
        super().__init__(parent)
        self._param_widgets: dict[str, QWidget] = {}
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        self._add_title(layout)
        self._add_filter_selector(layout)
        self._add_params_container(layout)
        self._add_apply_button(layout)

        # Initialize params for first filter
        self._update_params()

    def _add_title(self, layout: QVBoxLayout) -> None:
        """Add title label."""
        title = QLabel("Filter Configuration")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

    def _add_filter_selector(self, layout: QVBoxLayout) -> None:
        """Add filter type selector."""
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Filter Type:"))

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(list(FILTER_CONFIGS.keys()))
        selector_layout.addWidget(self.filter_combo)

        layout.addLayout(selector_layout)

    def _add_params_container(self, layout: QVBoxLayout) -> None:
        """Add container for filter parameters."""
        self.params_container = QWidget()
        self.params_layout = QFormLayout(self.params_container)
        layout.addWidget(self.params_container)

    def _add_apply_button(self, layout: QVBoxLayout) -> None:
        """Add apply button."""
        self.apply_button = QPushButton("Apply Filter")
        layout.addWidget(self.apply_button)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        self.apply_button.clicked.connect(self._on_apply_clicked)

    def _on_filter_changed(self, filter_type: str) -> None:
        """Handle filter type change."""
        self._update_params()

    def _on_apply_clicked(self) -> None:
        """Handle apply button click."""
        config = self.get_filter_config()
        self.filter_requested.emit(config)

    def _update_params(self) -> None:
        """Update parameter widgets for current filter type."""
        self._clear_params()
        filter_type = self.filter_combo.currentText()
        params = FILTER_CONFIGS.get(filter_type, {})

        for param_name, param_config in params.items():
            widget = self._create_param_widget(param_config)
            self._param_widgets[param_name] = widget
            self.params_layout.addRow(param_config[0] + ":", widget)

    def _clear_params(self) -> None:
        """Clear all parameter widgets."""
        self._param_widgets.clear()
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _create_param_widget(self, config: tuple[str, str, Any, Any, Any]) -> QWidget:
        """Create appropriate widget for parameter type."""
        _, param_type, default, min_val, max_val = config

        if param_type == "int":
            return self._create_int_spinbox(default, min_val, max_val)
        return self._create_float_spinbox(default, min_val, max_val)

    def _create_int_spinbox(self, default: int, min_val: int, max_val: int) -> QSpinBox:
        """Create integer spinbox."""
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        return spinbox

    def _create_float_spinbox(
        self, default: float, min_val: float, max_val: float
    ) -> QDoubleSpinBox:
        """Create float spinbox."""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setDecimals(4)
        spinbox.setSingleStep(0.01)
        return spinbox

    def get_filter_config(self) -> dict[str, Any]:
        """Get current filter configuration."""
        filter_type = self.filter_combo.currentText()
        parameters = self._get_param_values()

        return {
            "filter_type": filter_type,
            "parameters": parameters,
        }

    def _get_param_values(self) -> dict[str, Any]:
        """Get current parameter values."""
        values = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, QSpinBox | QDoubleSpinBox):
                values[name] = widget.value()
        return values

    def set_filter_type(self, filter_type: str) -> None:
        """Set the current filter type."""
        index = self.filter_combo.findText(filter_type)
        if index >= 0:
            self.filter_combo.setCurrentIndex(index)
