# ruff: noqa: E501
from typing import Any

"""Tests for AnalysisTab."""


import numpy as np
import pytest
from unittest.mock import MagicMock
from double_pendulum_golf.gui.analysis_tab import (
    AnalysisTab,
    _create_fallback_widget,
    _make_det_evaluator,
    _make_cond_evaluator,
)
import double_pendulum_golf.gui.analysis_tab as at

# ---------------------------------------------------------------------------
# GH1735 — DRY: unit tests for extracted evaluator factories
# ---------------------------------------------------------------------------


class TestMakeDetEvaluator:
    """_make_det_evaluator must return the determinant of the matrix returned by matrix_fn."""

    def test_det_of_identity(self) -> None:
        """det(I_2) = 1.0."""
        evaluator = _make_det_evaluator(lambda angles: np.eye(2))
        assert evaluator({}) == pytest.approx(1.0)

    def test_det_of_known_matrix(self) -> None:
        """det([[2,0],[0,3]]) = 6.0."""
        evaluator = _make_det_evaluator(lambda angles: np.array([[2.0, 0.0], [0.0, 3.0]]))
        assert evaluator({}) == pytest.approx(6.0)

    def test_det_passes_angles_to_fn(self) -> None:
        """matrix_fn receives the angles dict so it can use angle values."""
        received = {}

        def matrix_fn(angles: dict) -> np.ndarray:
            received.update(angles)
            return np.eye(2)

        evaluator = _make_det_evaluator(matrix_fn)
        evaluator({"phi": 1.5})
        assert received == {"phi": 1.5}


class TestMakeCondEvaluator:
    """_make_cond_evaluator must return the condition number of the matrix returned by matrix_fn."""

    def test_cond_of_identity(self) -> None:
        """cond(I_2) = 1.0."""
        evaluator = _make_cond_evaluator(lambda angles: np.eye(2))
        assert evaluator({}) == pytest.approx(1.0, rel=1e-6)

    def test_cond_of_diagonal(self) -> None:
        """cond(diag(1, 10)) = 10.0."""
        evaluator = _make_cond_evaluator(lambda angles: np.array([[1.0, 0.0], [0.0, 10.0]]))
        assert evaluator({}) == pytest.approx(10.0, rel=1e-6)

    def test_cond_passes_angles_to_fn(self) -> None:
        """matrix_fn receives the angles dict."""
        received = {}

        def matrix_fn(angles: dict) -> np.ndarray:
            received.update(angles)
            return np.eye(2)

        evaluator = _make_cond_evaluator(matrix_fn)
        evaluator({"theta1": 0.3})
        assert received == {"theta1": 0.3}


def test_analysis_tab_no_mpl(qapp, monkeypatch):
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


def test_analysis_tab_shared_evaluator_helpers(qapp) -> Any:
    tab = AnalysisTab()

    matrix_eval = tab._matrix_metric_evaluator(
        lambda angles: np.array([[angles["x"], 0.0], [0.0, 2.0]]),
        np.linalg.det,
    )
    assert matrix_eval({"x": 3.0}) == 6.0

    transformed_eval = tab._transformed_scalar_evaluator(
        lambda angles: np.array([angles["x"], angles["y"]]),
        np.sum,
    )
    assert transformed_eval({"x": 1.5, "y": 2.5}) == 4.0

    q_eval = tab._q_scalar_evaluator(
        lambda angles: np.array([angles["x"], angles["y"]]),
        np.linalg.norm,
    )
    assert q_eval({"x": 3.0, "y": 4.0}) == 5.0


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

    monkeypatch.setattr("double_pendulum_golf.data_extractor.extract_series", mock_extract)
    tab._on_plot_2d()
