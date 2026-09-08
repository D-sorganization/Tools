"""Tests for the ChartDataDialog dialog."""

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 not installed")
pytest.importorskip("pytestqt", reason="pytest-qt required for widget tests")

from PyQt6.QtWidgets import QDialogButtonBox

from double_pendulum_golf.gui.chart_data_dialog import ChartDataDialog


class TestChartDataDialog:
    def test_init_sets_up_ui(self, qtbot):
        dialog = ChartDataDialog(model_type="double")
        qtbot.addWidget(dialog)

        assert dialog.windowTitle() == "Select Chart Data"
        assert dialog._x_combo.count() > 0
        assert dialog._y_combo.count() > 0

    def test_init_raises_if_model_type_none(self):
        with pytest.raises((AssertionError, ValueError)):
            ChartDataDialog(model_type=None)  # type: ignore[attr-defined]

    def test_default_selections_are_set(self, qtbot):
        dialog = ChartDataDialog(model_type="double")
        qtbot.addWidget(dialog)

        assert dialog._x_combo.currentData() == "time"
        assert dialog._y_combo.currentData() == "tip_speed"
        assert dialog._reg_degree.value() == 3

    def test_get_selection_returns_values(self, qtbot):
        dialog = ChartDataDialog(model_type="double")
        qtbot.addWidget(dialog)

        x_idx = dialog._x_combo.findData("theta1")
        if x_idx >= 0:
            dialog._x_combo.setCurrentIndex(x_idx)

        y_idx = dialog._y_combo.findData("kinetic_energy")
        if y_idx >= 0:
            dialog._y_combo.setCurrentIndex(y_idx)

        dialog._reg_degree.setValue(5)

        x_key, y_key, degree = dialog.get_selection()
        assert x_key == "theta1"
        assert y_key == "kinetic_energy"
        assert degree == 5

    def test_buttons_connected(self, qtbot):
        dialog = ChartDataDialog(model_type="double")
        qtbot.addWidget(dialog)

        buttons = dialog.findChild(QDialogButtonBox)

        # Click OK
        with qtbot.waitSignal(dialog.accepted):
            buttons.button(QDialogButtonBox.StandardButton.Ok).click()

        dialog.show()

        # Click Cancel
        with qtbot.waitSignal(dialog.rejected):
            buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
