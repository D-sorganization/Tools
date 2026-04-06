from typing import Any

"""Tests for OptimizationWidget and its underlying logic."""


import numpy as np
from unittest.mock import MagicMock, patch

from double_pendulum_golf.gui.optimization_widget import (
    OptimizationWidget,
    _OptimizerWorker,
    CMAESState,
    _cmaes_step,
)


def dummy_objective(x) -> Any:
    # simple quadratic objective
    return np.sum(x**2)


def test_cmaes_step() -> Any:
    rng = np.random.default_rng(42)
    state = CMAESState(
        mean=np.array([1.0, 1.0]),
        sigma=1.0,
        C=np.eye(2),
        p_sigma=np.zeros(2),
        p_c=np.zeros(2),
    )

    new_state, fitnesses = _cmaes_step(state, dummy_objective, pop_size=10, rng=rng)
    assert new_state.generation == 1
    assert len(fitnesses) == 10
    assert fitnesses[0] <= fitnesses[-1]  # sorted

    # Test flat-line / exception in objective wrapper
    def bad_obj(x) -> Any:
        raise ValueError("bad")

    new_state_bad, fit_bad = _cmaes_step(state, bad_obj, pop_size=10, rng=rng)
    assert fit_bad[0] == float("inf")


def test_worker_cmaes(qapp) -> Any:
    worker = _OptimizerWorker(
        objective_fn=dummy_objective,
        n_params=2,
        n_iterations=5,
        method="CMA-ES",
        plateau_patience=2,
    )

    # Listen to signals
    fin_mock = MagicMock()
    worker.finished.connect(fin_mock)

    worker.run()

    fin_mock.assert_called_once()
    result = fin_mock.call_args[0][0]
    assert result["method"] == "CMA-ES"
    assert "coeffs" in result


def test_worker_de(qapp) -> Any:
    worker = _OptimizerWorker(
        objective_fn=dummy_objective,
        n_params=2,
        n_iterations=2,  # very fast
        method="Differential Evolution",
    )
    fin_mock = MagicMock()
    worker.finished.connect(fin_mock)
    worker.run()
    fin_mock.assert_called_once()

    # Cancel mid-run
    worker = _OptimizerWorker(
        objective_fn=dummy_objective,
        n_params=2,
        n_iterations=10,
        method="Differential Evolution",
    )
    fin_mock = MagicMock()
    worker.finished.connect(fin_mock)

    def on_iter(*args) -> Any:
        worker.cancel()

    worker.iteration_done.connect(on_iter)
    worker.run()


def test_worker_scipy_methods(qapp) -> Any:
    for method in ["Nelder-Mead", "L-BFGS-B"]:
        worker = _OptimizerWorker(
            objective_fn=dummy_objective,
            n_params=2,
            n_iterations=2,
            method=method,
            warm_start=np.array([0.5, 0.5]),  # Test warm start
        )
        fin_mock = MagicMock()
        worker.finished.connect(fin_mock)
        worker.run()
        fin_mock.assert_called_once()


def test_worker_error(qapp) -> Any:
    def error_obj(x) -> Any:
        raise RuntimeError("simulated error")

    worker = _OptimizerWorker(
        objective_fn=error_obj,
        n_params=2,
        n_iterations=5,
        method="Nelder-Mead",
    )
    err_mock = MagicMock()
    worker.error.connect(err_mock)
    worker.run()
    err_mock.assert_called_once()


@patch("double_pendulum_golf.gui.optimization_widget.QThread.start")
def test_optimization_widget_ui(mock_start, qapp) -> Any:
    w = OptimizationWidget("Test Model", 2)

    # Without objective fn
    w._on_run()
    assert "No objective function set" in w._log.toPlainText()

    # With objective fn
    w.set_objective_function(dummy_objective)
    w._cmb_method.setCurrentText("Nelder-Mead")
    w._spin_iters.setValue(5)

    # Trigger run
    w._on_run()
    mock_start.assert_called_once()

    # Simulate finished signal manually (since thread isn't truly starting)
    w._on_finished(
        {
            "speed": 10.0,
            "success": True,
            "method": "Nelder-Mead",
            "coeffs": [1.0, 1.0],
            "message": "done",
        }
    )

    # Next run warm started
    w._on_run()
    w._on_finished(
        {
            "speed": 10.0,
            "success": True,
            "method": "Nelder-Mead",
            "coeffs": [1.0, 1.0],
            "message": "done",
            "history": [10.0],
        }
    )

    w._on_apply()

    # Simulate iteration
    w._on_iteration(1, 10.0)

    # Try error
    w._on_error("Mock error")

    # Real run that gets cancelled
    w._spin_iters.setValue(100)
    w._on_run()
    w._on_cancel()

    # Clean up what happens when cancel is pressed
    # the worker internally receives cancel
    assert w._worker._cancelled is True


def test_optimization_widget_bound_objective_builder(qapp) -> Any:
    w = OptimizationWidget("Test Model", 2)
    params_getter = MagicMock(return_value={"gain": 3.0})
    objective_builder = MagicMock(return_value=dummy_objective)

    w.bind_objective_builder(params_getter, objective_builder)
    assert w._refresh_bound_objective() is True

    params_getter.assert_called_once()
    objective_builder.assert_called_once_with({"gain": 3.0})
    assert w._objective_fn is dummy_objective


def test_optimization_widget_bound_objective_builder_error(qapp) -> Any:
    w = OptimizationWidget("Test Model", 2)
    params_getter = MagicMock(side_effect=ValueError("bad params"))
    objective_builder = MagicMock()

    w.bind_objective_builder(params_getter, objective_builder)

    assert w._refresh_bound_objective() is False
    assert "Cannot build objective" in w._log.toPlainText()
