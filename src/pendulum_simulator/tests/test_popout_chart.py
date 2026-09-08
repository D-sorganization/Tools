from typing import Any
import numpy as np
import pytest
from unittest.mock import patch

from double_pendulum_golf.gui.popout_chart import PopOutChart, fit_regression


def test_fit_regression() -> Any:
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 4, 9])

    x_fit, y_fit, coeffs = fit_regression(x, y, 2)
    assert len(x_fit) == 200
    assert len(y_fit) == 200
    assert len(coeffs) == 3

    # Check assertion error
    with pytest.raises((ValueError, TypeError)):
        fit_regression(x, np.array([0, 1]), 2)

    with pytest.raises((ValueError, TypeError)):
        fit_regression(x, y, 11)


def test_popout_chart(qapp) -> Any:
    chart = PopOutChart()

    # Pre-plot regression should return None
    assert chart.add_regression() is None

    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 4, 9])

    # Plot data
    chart.plot_data(x, y, "xlabel", "ylabel", "title")
    assert chart._xlabel == "xlabel"
    assert chart._ylabel == "ylabel"
    assert chart._title == "title"

    # Add regression
    chart.add_regression(2)
    assert chart._regression is not None

    # Add regression with no non-zero coeffs (flat line)
    chart.plot_data(x, np.zeros_like(x), "x", "y", "t")
    chart.add_regression(1)

    # Show without mpl
    with patch("double_pendulum_golf.gui.popout_chart._HAS_MPL", False):
        with patch("PyQt6.QtWidgets.QMessageBox.information") as mock_info:
            chart.show()
            mock_info.assert_called_once()

    # Show with mpl
    with patch("double_pendulum_golf.gui.popout_chart._HAS_MPL", True):
        with (
            patch("double_pendulum_golf.gui.popout_chart.Figure"),
            patch("double_pendulum_golf.gui.popout_chart.FigureCanvasQTAgg"),
            patch("PyQt6.QtWidgets.QMainWindow.show"),
            patch("PyQt6.QtWidgets.QMainWindow.setWindowTitle"),
            patch("PyQt6.QtWidgets.QMainWindow.setMinimumSize"),
            patch("PyQt6.QtWidgets.QMainWindow.setAttribute"),
            patch("PyQt6.QtWidgets.QMainWindow.setCentralWidget"),
            patch("PyQt6.QtWidgets.QVBoxLayout.addWidget"),
        ):
            chart.show()
            assert chart._fig is not None
            assert chart._ax is not None
            assert chart._window is not None

    # Show with no data
    chart2 = PopOutChart()
    with patch("double_pendulum_golf.gui.popout_chart._HAS_MPL", True):
        chart2.show()
        assert chart2._window is None
