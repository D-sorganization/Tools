"""Reusable port editing widgets for the vessel drafter GUI."""

from __future__ import annotations

from dataclasses import dataclass

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
    def __init__(self, title: str, headers: tuple[str, str, str]):
        super().__init__(title)
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(list(headers))
        header = self.table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        self.add_button = QPushButton("Add Port")
        self.remove_button = QPushButton("Remove Selected")

        button_row = QHBoxLayout()
        button_row.addWidget(self.add_button)
        button_row.addWidget(self.remove_button)

        root = QVBoxLayout(self)
        root.addWidget(self.table)
        root.addLayout(button_row)

    def append_row(self, values: tuple[float, float, float]) -> None:
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)
        for column_index, value in enumerate(values):
            self.table.setItem(
                row_index,
                column_index,
                QTableWidgetItem(f"{value:.2f}"),
            )

    def rows(self) -> tuple[tuple[float, float, float], ...]:
        items: list[tuple[float, float, float]] = []
        for row_index in range(self.table.rowCount()):
            row = []
            for column_index in range(self.table.columnCount()):
                item = self.table.item(row_index, column_index)
                row.append(0.0 if item is None else float(item.text()))
            items.append((row[0], row[1], row[2]))
        return tuple(items)

    def remove_selected_rows(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.table.selectedIndexes()},
            reverse=True,
        )
        for row_index in selected_rows:
            self.table.removeRow(row_index)

    def set_rows(self, rows: tuple[tuple[float, float, float], ...]) -> None:
        self.table.setRowCount(0)
        for row in rows:
            self.append_row(row)
