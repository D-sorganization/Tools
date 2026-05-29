"""Unit tests for ``process_calculators.optimization``.

Covers the surface-optimization entry point ``find_optimal_on_surface`` (all
three solver methods plus the validation/error branches) and the Adam optimizer
``run_adam_optimization`` with a stubbed ``evaluate_output`` so no real
gasification engine is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from sidekick.process_calculators import optimization as opt

# ---------------------------------------------------------------------------
# find_optimal_on_surface
# ---------------------------------------------------------------------------


def _paraboloid(n: int = 11):
    """Return (x, y, Z) for Z = -((x-0.6)^2 + (y-0.3)^2); max near (0.6, 0.3)."""
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)  # shape (len(y), len(x))
    z = -((xx - 0.6) ** 2 + (yy - 0.3) ** 2)
    return x, y, z


def test_grid_search_finds_maximum_region() -> None:
    x, y, z = _paraboloid()
    result = opt.find_optimal_on_surface(x, y, z, method="Grid Search")
    assert result["success"] is True
    assert result["optimal_x"] == pytest.approx(0.6, abs=0.15)
    assert result["optimal_y"] == pytest.approx(0.3, abs=0.15)
    assert result["evaluations"] == 225  # 15x15 grid


def test_grid_search_respects_explicit_bounds() -> None:
    x, y, z = _paraboloid()
    result = opt.find_optimal_on_surface(
        x, y, z, method="Grid Search", bounds=((0.0, 0.5), (0.0, 0.5))
    )
    assert 0.0 <= result["optimal_x"] <= 0.5
    assert 0.0 <= result["optimal_y"] <= 0.5


def test_lbfgsb_method_runs() -> None:
    x, y, z = _paraboloid()
    result = opt.find_optimal_on_surface(x, y, z, method="L-BFGS-B")
    assert "optimal_z" in result
    assert result["evaluations"] >= 1


def test_differential_evolution_method_runs() -> None:
    x, y, z = _paraboloid()
    result = opt.find_optimal_on_surface(x, y, z, method="Differential Evolution")
    assert result["optimal_x"] == pytest.approx(0.6, abs=0.2)
    assert result["evaluations"] >= 1


def test_callback_invoked_during_grid_search() -> None:
    x, y, z = _paraboloid()
    calls: list[tuple[int, int]] = []
    opt.find_optimal_on_surface(
        x, y, z, method="Grid Search", callback=lambda c, t: calls.append((c, t))
    )
    assert calls, "callback should have been invoked at least once"
    assert all(total == 225 for _, total in calls)


def test_accepts_2d_grids() -> None:
    x, y, z = _paraboloid()
    xx, yy = np.meshgrid(x, y)
    result = opt.find_optimal_on_surface(xx, yy, z, method="Grid Search")
    assert result["success"] is True


def test_transposed_z_shape_accepted() -> None:
    x = np.linspace(0.0, 1.0, 7)
    y = np.linspace(0.0, 1.0, 5)
    xx, yy = np.meshgrid(x, y, indexing="ij")  # shape (len(x), len(y))
    z = -((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    result = opt.find_optimal_on_surface(x, y, z, method="Grid Search")
    assert result["success"] is True


def test_unknown_method_raises() -> None:
    x, y, z = _paraboloid()
    with pytest.raises(ValueError, match="Unknown optimization method"):
        opt.find_optimal_on_surface(x, y, z, method="nope")


def test_mismatched_z_shape_raises() -> None:
    x = np.linspace(0.0, 1.0, 7)
    y = np.linspace(0.0, 1.0, 5)
    z = np.zeros((3, 3))  # neither (5,7) nor (7,5)
    with pytest.raises(ValueError, match="does not match"):
        opt.find_optimal_on_surface(x, y, z, method="Grid Search")


def test_none_x_grid_raises() -> None:
    _, y, z = _paraboloid()
    with pytest.raises(ValueError, match="x_grid must be provided"):
        opt.find_optimal_on_surface(None, y, z)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_adam_optimization
# ---------------------------------------------------------------------------


def _patch_quadratic_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub evaluate_output with a smooth concave function of Temperature.

    Objective peaks at Temperature == 700, so a maximizing run should climb.
    """

    def fake_evaluate_output(engine, base_params, manual_hhv, output_name, overrides):
        temp = overrides.get("Temperature", 0.0)
        value = -((temp - 700.0) ** 2) / 1.0e4
        return value, {"H2": 0.5}, {"state": temp}

    monkeypatch.setattr(opt, "evaluate_output", fake_evaluate_output)


def test_run_adam_optimization_climbs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_quadratic_objective(monkeypatch)
    configs = [{"name": "Temperature", "min": 500.0, "max": 900.0, "initial": 600.0}]
    analysis_params = {"base_params": {}, "output_variable": "HHV"}

    result = opt.run_adam_optimization(
        engine=object(),
        analysis_params=analysis_params,
        manual_hhv=0.0,
        parameter_configs=configs,
        maximize=True,
        learning_rate=10.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        gradient_step=1.0,
        max_iterations=40,
        tolerance=1e-9,
    )

    assert set(result) == {
        "best_output",
        "best_parameters",
        "best_state",
        "best_composition",
        "history",
        "final_parameters",
        "iterations",
    }
    assert result["iterations"] >= 1
    assert result["best_parameters"]["Temperature"] == pytest.approx(700.0, abs=60.0)
    assert result["best_output"] <= 0.0


def test_run_adam_optimization_requires_configs() -> None:
    with pytest.raises(ValueError, match="At least one parameter"):
        opt.run_adam_optimization(
            engine=object(),
            analysis_params={"base_params": {}, "output_variable": "HHV"},
            manual_hhv=0.0,
            parameter_configs=[],
            maximize=True,
            learning_rate=1.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_step=1.0,
            max_iterations=5,
            tolerance=1e-6,
        )


def test_build_override_mapping_filters_unknown_names() -> None:
    override = opt._build_override_mapping(
        ["Temperature", "Bogus", "Pressure"], [700.0, 1.0, 5.0]
    )
    assert override == {"Temperature": 700.0, "Pressure": 5.0}


def test_build_override_mapping_none_raises() -> None:
    with pytest.raises(ValueError, match="parameter_names must be provided"):
        opt._build_override_mapping(None, [1.0])  # type: ignore[arg-type]
