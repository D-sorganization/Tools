"""Focused neural-surrogate training controls for the PyQt laboratory."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class NeuralTrainingControls(QWidget):
    """Collect training variables without owning process or data behavior."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._create_fields()
        self._build_layout()

    @staticmethod
    def _help(widget: QWidget, name: str, tip: str) -> None:
        widget.setAccessibleName(name)
        widget.setToolTip(tip)

    def _create_fields(self) -> None:
        self.vendor_combo = QComboBox()
        self.add_vendor(
            "TrackMan-comparable",
            "Requires shot-level TrackMan outcomes in the selected dataset.",
        )
        self.add_vendor(
            "Foresight-comparable",
            "No reusable shot-level Foresight outcome corpus is available.",
        )
        self.add_vendor(
            "FlightScope-comparable",
            "No reusable shot-level FlightScope outcome corpus is available.",
        )
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.target_list = QListWidget()
        self.target_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.split_group_combo = QComboBox()
        self.hidden_layers_edit = QLineEdit("64, 32")
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["relu", "tanh"])
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setDecimals(7)
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setValue(0.0001)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1_000_000)
        self.epochs_spin.setValue(500)
        self.holdout_spin = QDoubleSpinBox()
        self.holdout_spin.setRange(0.05, 0.5)
        self.holdout_spin.setSingleStep(0.05)
        self.holdout_spin.setValue(0.2)
        self.train_button = QPushButton("Train in Private Campaign...")
        self.cancel_button = QPushButton("Cancel Training")
        self.cancel_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

    def _build_layout(self) -> None:
        form = QFormLayout()
        for label, widget, tip in self._fields():
            self._help(widget, label, tip)
            form.addRow(label + ":", widget)
        self._help(
            self.train_button,
            "Start Training",
            "Write an inspectable request and asynchronously launch the private CLI",
        )
        self._help(
            self.cancel_button,
            "Cancel Training",
            "Terminate the currently running private training process",
        )
        self._help(
            self.progress,
            "Training Progress",
            "Private CLI progress; indeterminate until the process exits",
        )
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.train_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.progress)

    def _fields(self) -> tuple[tuple[str, QWidget, str], ...]:
        return (
            (
                "Vendor target",
                self.vendor_combo,
                "Choose an evidence-supported vendor-comparable target",
            ),
            (
                "Input features",
                self.feature_list,
                "Select numeric model inputs; avoid outcome leakage",
            ),
            (
                "Output targets",
                self.target_list,
                "Select one or more measured shot outcomes",
            ),
            (
                "Split group",
                self.split_group_combo,
                "Keep each identity wholly in one data split",
            ),
            (
                "Hidden layers",
                self.hidden_layers_edit,
                "Comma-separated neuron counts, such as 64, 32",
            ),
            (
                "Activation",
                self.activation_combo,
                "Activation for hidden neural layers",
            ),
            (
                "L2 alpha",
                self.alpha_spin,
                "Regularization strength; larger values shrink weights",
            ),
            ("Random seed", self.seed_spin, "Fixed split and initialization seed"),
            ("Epoch limit", self.epochs_spin, "Maximum training passes"),
            (
                "Holdout fraction",
                self.holdout_spin,
                "Rows reserved for validation and independent testing",
            ),
        )

    def add_vendor(self, label: str, reason: str) -> None:
        """Add a disabled vendor with its evidence limitation as a tooltip."""

        self.vendor_combo.addItem(label, label)
        index = self.vendor_combo.count() - 1
        model = self.vendor_combo.model()
        assert isinstance(model, QStandardItemModel)
        item = model.item(index)
        assert item is not None
        item.setEnabled(False)
        self.vendor_combo.setItemData(index, reason, Qt.ItemDataRole.ToolTipRole)

    def interactive_controls(self) -> tuple[QWidget, ...]:
        """Return all interactive controls for accessibility tests."""

        return (
            self.vendor_combo,
            self.feature_list,
            self.target_list,
            self.split_group_combo,
            self.hidden_layers_edit,
            self.activation_combo,
            self.alpha_spin,
            self.seed_spin,
            self.epochs_spin,
            self.holdout_spin,
            self.train_button,
            self.cancel_button,
            self.progress,
        )


__all__ = ["NeuralTrainingControls"]
