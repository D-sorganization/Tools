from typing import Any

"""Tests for FunctionGeneratorDialog."""

import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QDialogButtonBox

from double_pendulum_golf.gui.function_generator_dialog import FunctionGeneratorDialog


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
