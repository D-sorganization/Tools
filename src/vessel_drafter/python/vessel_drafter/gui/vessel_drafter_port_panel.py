"""Reusable port editing widgets for the vessel drafter GUI."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def make_double_spin(value: float, minimum: float, maximum: float) -> QDoubleSpinBox:
    if not (value is not None):
        raise ValueError("value must be provided")
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(2)
    spin.setSingleStep(0.5)
    spin.setValue(value)
    return spin


@dataclass(frozen=True)
class PortFieldSpec:
    label: str
    default: float
    minimum: float
    maximum: float


class PortValueDialog(QDialog):
    def __init__(self, title: str, fields: tuple[PortFieldSpec, ...], parent: QWidget):
        if not (title is not None):
            raise ValueError("title must be provided")
        super().__init__(parent)
        self.setWindowTitle(title)
        self._spins = tuple(
            make_double_spin(field.default, field.minimum, field.maximum)
            for field in fields
        )

        form_layout = QFormLayout()
        for field, spin in zip(fields, self._spins, strict=True):
            form_layout.addRow(field.label, spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form_layout)
        root.addWidget(buttons)

    def values(self) -> tuple[float, ...]:
        return tuple(spin.value() for spin in self._spins)


class PortTableSection(QGroupBox):
    """Panel containing an editable port table with Add / Remove buttons.

    LOD boundary
    ------------
    External code must not access ``add_button``, ``remove_button``, or
    ``table`` directly.  Instead, connect to the signals exposed below and
    call the public methods to mutate state.
    """

    # Emitted when the "Add Port" button is clicked.
    add_requested: pyqtSignal = pyqtSignal()
    # Emitted when the "Remove Selected" button is clicked.
    remove_requested: pyqtSignal = pyqtSignal()
    # Emitted whenever a cell value changes in the table.
    data_changed: pyqtSignal = pyqtSignal()

    def __init__(self, title: str, headers: tuple[str, str, str]):
        if not (title is not None):
            raise ValueError("title must be provided")
        super().__init__(title)
        self._table = QTableWidget(0, len(headers))
        self._table.setHorizontalHeaderLabels(list(headers))
        header = self._table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        self._add_button = QPushButton("Add Port")
        self._remove_button = QPushButton("Remove Selected")

        # Forward button clicks to public signals so callers never touch internals.
        self._add_button.clicked.connect(self.add_requested)
        self._remove_button.clicked.connect(self.remove_requested)
        self._table.itemChanged.connect(self.data_changed)

        button_row = QHBoxLayout()
        button_row.addWidget(self._add_button)
        button_row.addWidget(self._remove_button)

        root = QVBoxLayout(self)
        root.addWidget(self._table)
        root.addLayout(button_row)

    def append_row(self, values: tuple[float, float, float]) -> None:
        if not (values is not None):
            raise ValueError("values must be provided")
        row_index = self._table.rowCount()
        self._table.insertRow(row_index)
        for column_index, value in enumerate(values):
            self._table.setItem(
                row_index,
                column_index,
                QTableWidgetItem(f"{value:.2f}"),
            )

    def rows(self) -> tuple[tuple[float, float, float], ...]:
        items: list[tuple[float, float, float]] = []
        for row_index in range(self._table.rowCount()):
            row = []
            for column_index in range(self._table.columnCount()):
                item = self._table.item(row_index, column_index)
                row.append(0.0 if item is None else float(item.text()))
            items.append((row[0], row[1], row[2]))
        return tuple(items)

    def remove_selected_rows(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self._table.selectedIndexes()},
            reverse=True,
        )
        for row_index in selected_rows:
            self._table.removeRow(row_index)

    def set_rows(self, rows: tuple[tuple[float, float, float], ...]) -> None:
        self._table.setRowCount(0)
        for row in rows:
            self.append_row(row)
