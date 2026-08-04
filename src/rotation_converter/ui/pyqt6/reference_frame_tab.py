# ruff: noqa: E501
"""PyQt6 tab for educational reference-frame conversion operations."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rotation_converter.reference_frame_operations import (
    OperationName,
    compute_reference_frame_operation,
)

_OPS: tuple[tuple[str, OperationName], ...] = (
    ("Twist Frame Conversion (Adjoint)", "twist_frame_conversion"),
    ("Homogeneous Transform Builder", "homogeneous_transform"),
    ("so(3) <-> SO(3) Exponential/Log", "so3_so3_maps"),
)


def _parse_numbers(value: str) -> np.ndarray:
    parts = value.replace(",", " ").split()
    return cast(np.ndarray, np.asarray([float(part) for part in parts], dtype=float))


def _make_matrix_grid(parent: QWidget, rows: int, cols: int) -> list[list[QLineEdit]]:
    if parent is None:
        raise ValueError("parent must be provided")
    layout = QGridLayout(parent)
    edits: list[list[QLineEdit]] = []
    for i in range(rows):
        row: list[QLineEdit] = []
        for j in range(cols):
            edit = QLineEdit("0")
            layout.addWidget(edit, i, j)
            row.append(edit)
        edits.append(row)
    return edits


def _read_matrix(edits: list[list[QLineEdit]]) -> np.ndarray:
    rows = []
    rows.extend([[float(edit.text()) for edit in row_edits] for row_edits in edits])
    return cast(np.ndarray, np.asarray(rows, dtype=float))


class ReferenceFrameTab(QWidget):
    """Educational reference-frame operation tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._on_operation_changed()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left_group = QGroupBox("Reference-Frame Operations")
        left_layout = QVBoxLayout(left_group)
        left_form = QFormLayout()

        self._operation = QComboBox()
        for label, _ in _OPS:
            self._operation.addItem(label)
        left_form.addRow("Operation:", self._operation)

        self._transform_group = QGroupBox("Transform T (4x4)")
        self._transform_edits = _make_matrix_grid(self._transform_group, 4, 4)
        for i in range(4):
            self._transform_edits[i][i].setText("1")

        self._twist_input = QLineEdit("0 0 1 0.5 0 0")
        left_form.addRow("Twist [wx wy wz vx vy vz]:", self._twist_input)

        self._rotation_group = QGroupBox("Rotation Matrix R (3x3)")
        self._rotation_edits = _make_matrix_grid(self._rotation_group, 3, 3)
        for i in range(3):
            self._rotation_edits[i][i].setText("1")

        self._translation_input = QLineEdit("0 0 0")
        left_form.addRow("Translation p [px py pz]:", self._translation_input)

        self._so3_vector_input = QLineEdit("0 0 0.5")
        left_form.addRow("so(3) vector [wx wy wz]:", self._so3_vector_input)

        self._compute_button = QPushButton("Compute")
        left_form.addRow(self._compute_button)
        left_layout.addLayout(left_form)
        left_layout.addWidget(self._transform_group)
        left_layout.addWidget(self._rotation_group)
        root.addWidget(left_group, 1)

        right_group = QGroupBox("Educational Output")
        right_layout = QVBoxLayout(right_group)
        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Results JSON will appear here.")
        self._markdown = QTextEdit()
        self._markdown.setReadOnly(True)
        self._markdown.setPlaceholderText("Explanation (markdown) will appear here.")
        self._latex = QTextEdit()
        self._latex.setReadOnly(True)
        self._latex.setPlaceholderText("Formulas (latex) will appear here.")
        right_layout.addWidget(QLabel("Results (JSON)"))
        right_layout.addWidget(self._results, 2)
        right_layout.addWidget(QLabel("Explanation (Markdown)"))
        right_layout.addWidget(self._markdown, 2)
        right_layout.addWidget(QLabel("Formulas (LaTeX)"))
        right_layout.addWidget(self._latex, 2)
        root.addWidget(right_group, 1)

        self._operation.currentIndexChanged.connect(self._on_operation_changed)
        self._compute_button.clicked.connect(self._compute)

    def _current_operation(self) -> OperationName:
        return _OPS[self._operation.currentIndex()][1]

    def _on_operation_changed(self) -> None:
        operation = self._current_operation()
        self._transform_group.setVisible(operation == "twist_frame_conversion")
        self._twist_input.setVisible(operation == "twist_frame_conversion")
        self._rotation_group.setVisible(operation == "homogeneous_transform")
        self._translation_input.setVisible(operation == "homogeneous_transform")
        self._so3_vector_input.setVisible(operation == "so3_so3_maps")

    def _compute(self) -> None:
        operation = self._current_operation()
        try:
            result = self._compute_operation(operation)
            self._results.setPlainText(json.dumps(result.results, indent=2))
            self._markdown.setPlainText(result.explanation_markdown)
            self._latex.setPlainText(result.explanation_latex)
        except (
            Exception
        ) as error:  # noqa: BLE001 — user input can raise any error; display it
            self._results.setPlainText(f"Error: {error}")
            self._markdown.clear()
            self._latex.clear()

    def _compute_operation(self, operation: OperationName) -> Any:
        if operation is None:
            raise ValueError("operation must be provided")
        if operation == "twist_frame_conversion":
            transform = _read_matrix(self._transform_edits)
            twist = _parse_numbers(self._twist_input.text())
            return compute_reference_frame_operation(
                operation, transform=transform, twist=twist
            )
        if operation == "homogeneous_transform":
            rotation = _read_matrix(self._rotation_edits)
            translation = _parse_numbers(self._translation_input.text())
            return compute_reference_frame_operation(
                operation, rotation_matrix=rotation, translation=translation
            )
        so3_vector = _parse_numbers(self._so3_vector_input.text())
        return compute_reference_frame_operation(operation, so3_vector=so3_vector)
