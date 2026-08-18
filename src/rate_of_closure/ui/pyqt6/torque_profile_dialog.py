"""Polynomial-generator dialog for prescribed joint-torque profiles."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QLabel, QMessageBox, QVBoxLayout, QWidget

from shared.python.signal_toolkit.polynomial_generator import (
    PolynomialGeneratorWidget,
)


class TorquePolynomialDialog(QDialog):
    """Modal host for the shared polynomial generator."""

    polynomialAccepted = pyqtSignal(str, list)  # noqa: N815

    def __init__(self, joint_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Design Torque — {joint_id}")
        self.resize(1000, 760)
        layout = QVBoxLayout(self)
        description = QLabel(
            "Draw, enter, or fit torque versus physical time. Generated "
            "coefficients are stored as [c0, c1, …] in N·m."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.generator = PolynomialGeneratorWidget(
            self,
            use_builtin_theme=False,
            error_handler=self._show_error,
        )
        self.generator.set_joints([joint_id])
        self.generator.polynomial_generated.connect(self._accept_polynomial)
        layout.addWidget(self.generator)

    def _accept_polynomial(self, joint_id: str, coefficients: list[float]) -> None:
        self.polynomialAccepted.emit(joint_id, coefficients)
        self.accept()

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)


__all__ = ["TorquePolynomialDialog"]
