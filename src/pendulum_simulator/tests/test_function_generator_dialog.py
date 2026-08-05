from typing import Any

"""Tests for FunctionGeneratorDialog."""


import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QDialogButtonBox

from double_pendulum_golf.gui.function_generator_dialog import FunctionGeneratorDialog
from double_pendulum_golf.torque_utils import make_polynomial_torque


def test_function_generator_dialog_init(qapp) -> Any:
    dlg = FunctionGeneratorDialog(joint_names=["Shoulder", "Elbow"])
    assert dlg.windowTitle() == "Signal Toolkit — Torque Profile Designer"

    # Check if tabs were created (if widgets are available)
    import double_pendulum_golf.gui.function_generator_dialog as fgd

    if fgd._WIDGET_AVAILABLE:
        assert hasattr(dlg, "_poly_widget") or hasattr(dlg, "_signal_widget")

    btn_box = dlg.findChild(QDialogButtonBox)
    assert btn_box is not None


def test_on_signal_applied(qapp) -> Any:
    from unittest.mock import MagicMock

    dlg = FunctionGeneratorDialog()

    mock_emit = MagicMock()
    dlg.torque_imported.connect(mock_emit)

    dlg._on_signal_applied("Wrist", [1.0, 2.0, 3.0])

    # PyQt signals pass arguments to connected slot
    mock_emit.assert_called_once_with("wrist", [1.0, 2.0, 3.0])
    assert dlg.result() == 1  # QDialog.DialogCode.Accepted


def test_designed_polynomial_preserves_torque_semantics(qapp) -> None:
    """The visual fit and pendulum evaluator must use the same coefficient order."""
    dlg = FunctionGeneratorDialog(joint_names=["Shoulder"])
    if not hasattr(dlg, "_poly_widget"):
        pytest.skip("Polynomial generator widget is unavailable")

    emitted: list[tuple[str, list[float]]] = []
    dlg.torque_imported.connect(
        lambda joint, coeffs: emitted.append((joint, list(coeffs)))
    )
    widget = dlg._poly_widget
    widget.order_spin.setValue(2)
    widget.current_points = [
        (0.0, 2.0),
        (1.0, 9.0),
        (2.0, 24.0),
        (3.0, 47.0),
    ]  # tau(t) = 2 + 3 t + 4 t^2

    widget._fit_polynomial_or_raise()

    assert emitted[0][0] == "shoulder"
    torque = make_polynomial_torque(emitted[0][1])
    assert torque(2.5)[0] == pytest.approx(34.5)


def test_function_generator_no_widgets(qapp, monkeypatch) -> Any:
    import double_pendulum_golf.gui.function_generator_dialog as fgd

    monkeypatch.setattr(fgd, "_HAS_POLY_WIDGET", False)
    monkeypatch.setattr(fgd, "_HAS_SIGNAL_WIDGET", False)

    dlg = FunctionGeneratorDialog()
    # It should render the fallback text edit
    from PyQt6.QtWidgets import QTextEdit

    note = dlg.findChild(QTextEdit)
    assert note is not None
    assert "Signal Toolkit widgets not available" in note.toPlainText()
