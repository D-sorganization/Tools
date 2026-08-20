from __future__ import annotations

from PyQt6.QtWidgets import QPushButton

from rate_of_closure.ui.pyqt6.neural_model_lab_tab import NeuralModelLabTab


def test_neural_lab_is_capability_driven_accessible_and_fail_closed(
    qtbot: object,
) -> None:
    tab = NeuralModelLabTab()
    qtbot.addWidget(tab)
    text = tab.capability.toPlainText()
    assert "TrackMan: unavailable — 11,699 rows / 9,298 strict" in text
    assert "retired_non_group_safe" in text
    assert "Foresight: unavailable — 4 rows / 2 strict" in text
    assert "FlightScope: unavailable — 2,794 rows / 0 strict" in text
    assert (
        "Residual plot unavailable" in tab.residual_plot.toolTip()
        or tab.residual_plot.toolTip()
    )
    for button in tab.findChildren(QPushButton):
        assert button.accessibleName()
        assert button.toolTip()
