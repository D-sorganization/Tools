from typing import Any

"""Tests for AnalysisTab."""


import numpy as np
from unittest.mock import MagicMock
from double_pendulum_golf.gui.analysis_tab import AnalysisTab, _create_fallback_widget
import double_pendulum_golf.gui.analysis_tab as at


def test_analysis_tab_no_mpl(qapp, monkeypatch) -> Any:
    monkeypatch.setattr(at, "_HAS_MPL", False)

    # Check fallback UI
    fallback = _create_fallback_widget()
    assert "Install matplotlib" in fallback.text()

    tab = AnalysisTab()
    # these should return early
    tab._on_plot_2d()
    tab._on_plot_surface()


def test_analysis_tab_plotting(qapp, monkeypatch) -> Any:
    tab = AnalysisTab()

    # Mock result
    class MockResult:
        t = np.array([0.0, 1.0])
        states = [[0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.1, 0.1]]
        params = MagicMock()

    res = MockResult()
    tab.set_result(res, "double")

    # mock extract_series
    monkeypatch.setattr(
        "double_pendulum_golf.data_extractor.extract_series",
        lambda r, k, m: (np.array([0.0, 1.0]), "Desc", "Unit"),
    )

    tab._x_combo.setCurrentIndex(0)
    tab._y_combo.setCurrentIndex(1)

    tab._on_plot_2d()

    tab._reg_spin.setValue(1)
    # mock popout_chart fit_regression
    monkeypatch.setattr(
        "double_pendulum_golf.gui.popout_chart.fit_regression",
        lambda x, y, d: (x, y, "poly"),
    )
    tab._on_plot_2d()

    tab.plot_2d("time", "tip_speed")


def test_analysis_tab_surface(qapp, monkeypatch) -> Any:
    tab = AnalysisTab()
    res = MagicMock()
    del res.params  # Use default params!
    tab.set_result(res, "double")
    tab.widget()  # cover 190

    # Try identical axes
    tab._x3_combo.setCurrentIndex(0)
    tab._y3_combo.setCurrentIndex(0)
    tab._on_plot_surface()

    # Different axes
    tab._x3_combo.setCurrentIndex(0)
    tab._y3_combo.setCurrentIndex(1)
    tab._z3_combo.setCurrentIndex(0)  # det(M)
    tab._sweep_points.setValue(2)  # very small grid for speed

    # test double
    tab._on_plot_surface()

    # test double exception path LinAlgError
    tab._get_surface_evaluator = lambda z: (
        lambda q: np.linalg.cond(np.array([[0, 0], [0, 0]]))
    )
    tab._on_plot_surface()
    del tab._get_surface_evaluator  # restore

    # test unknown evaluator path
    tab._z3_combo.addItem("Unknown", "unknown")
    tab._z3_combo.setCurrentIndex(tab._z3_combo.count() - 1)
    tab._on_plot_surface()
    tab._z3_combo.setCurrentIndex(0)  # restore

    # test triple
    tab._model_type = "triple"
    tab._populate_surface_combos()
    tab._x3_combo.setCurrentIndex(0)
    tab._y3_combo.setCurrentIndex(1)
    tab._z3_combo.setCurrentIndex(1)  # cond(M)
    tab._on_plot_surface()

    # test golfer
    tab._model_type = "golfer"
    tab._populate_surface_combos()
    tab._x3_combo.setCurrentIndex(0)
    tab._y3_combo.setCurrentIndex(1)
    tab._z3_combo.setCurrentIndex(2)  # PE
    tab._on_plot_surface()

    # test empty result
    tab.set_result(None)
    tab._on_plot_2d()
    tab.plot_2d("time", "tip_speed")


def test_analysis_tab_evaluators(qapp) -> Any:
    tab = AnalysisTab()

    # test double
    d_eval = tab._get_surface_evaluator("mass_matrix_det")
    assert isinstance(d_eval({"theta1": 0.0}), float)
    c_eval = tab._get_surface_evaluator("mass_matrix_cond")
    assert isinstance(c_eval({"theta1": 0.0}), float)
    pe_eval = tab._get_surface_evaluator("potential_energy")
    assert isinstance(pe_eval({"theta1": 0.0, "phi": 0.0}), float)
    mani_eval = tab._get_surface_evaluator("manipulability")
    if mani_eval:
        assert isinstance(mani_eval({"theta1": 0.0, "phi": 0.0}), float)
    assert tab._get_surface_evaluator("unknown") is None

    # test triple
    tab._model_type = "triple"
    d_eval2 = tab._get_surface_evaluator("mass_matrix_det")
    assert isinstance(d_eval2({"theta1": 0.0}), float)
    pe_eval2 = tab._get_surface_evaluator("potential_energy")
    assert isinstance(pe_eval2({"theta1": 0.0}), float)
    mani_eval2 = tab._get_surface_evaluator("manipulability")
    if mani_eval2:
        assert isinstance(mani_eval2({"theta1": 0.0, "phi1": 0.0, "phi2": 0.0}), float)

    # test golfer
    tab._model_type = "golfer"
    d_eval3 = tab._get_surface_evaluator("mass_matrix_det")
    assert isinstance(d_eval3({"theta1": 0.0}), float)
    c_eval3 = tab._get_surface_evaluator("mass_matrix_cond")
    assert isinstance(c_eval3({"theta1": 0.0}), float)
    pe_eval3 = tab._get_surface_evaluator("potential_energy")
    assert isinstance(pe_eval3({"theta1": 0.0}), float)
    mani_eval3 = tab._get_surface_evaluator("manipulability")
    if mani_eval3:
        assert isinstance(mani_eval3({"theta1": 0.0}), float)


def test_analysis_tab_plot_2d_errors(qapp, monkeypatch) -> Any:
    tab = AnalysisTab()
    tab.set_result(MagicMock(), "double")

    # x_key is None
    tab._x_combo.clear()
    tab._y_combo.clear()
    tab._on_plot_2d()

    # KeyError in extract_series
    tab._x_combo.addItem("foo", "foo")
    tab._y_combo.addItem("bar", "bar")
    tab._x_combo.setCurrentIndex(0)
    tab._y_combo.setCurrentIndex(0)

    def mock_extract(*args) -> Any:
        raise KeyError()

    monkeypatch.setattr(
        "double_pendulum_golf.data_extractor.extract_series", mock_extract
    )
    tab._on_plot_2d()
