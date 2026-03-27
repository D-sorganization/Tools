from typing import Any

"""Tests for the mathematical equations popup window."""

import pytest
from PyQt6.QtWidgets import QDialog, QTextBrowser, QPushButton
from double_pendulum_golf.gui.equations_popup import show_equations_popup, EquationTopic


def test_show_equations_popup(qapp, qtbot) -> Any:
    # Test valid topic
    dlg = show_equations_popup(None, EquationTopic.MASS_MATRIX)

    assert isinstance(dlg, QDialog)
    assert dlg.windowTitle() == "Mass Matrix — Derivation"

    # Internal widgets
    browser = dlg.findChild(QTextBrowser)
    assert browser is not None
    assert "Mass (Inertia) Matrix" in browser.toHtml()

    copy_btn = dlg.findChild(QPushButton)
    assert copy_btn is not None
    assert copy_btn.text() == "Copy to Clipboard"

    # Test copy button
    with qtbot.waitSignal(copy_btn.clicked, timeout=1000, raising=False):
        copy_btn.click()

    cb = qapp.clipboard()
    assert "Mass (Inertia) Matrix" in cb.text()


def test_show_equations_popup_invalid_topic(qapp) -> Any:
    with pytest.raises(AssertionError):
        show_equations_popup(None, "INVALID_TOPIC")
