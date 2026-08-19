"""Cross-runtime resource authority for the managed Plots workspace."""

from __future__ import annotations

import dataclasses

import pytest

from rate_of_closure.plot_workspace_limits import (
    MAX_MANAGED_PLOTS,
    MAX_SWEEP_EVALUATIONS,
    plot_evaluation_count,
    validate_plot_workspace,
)
from rate_of_closure.plotting import builtin_spec


def _sweep(count: int):  # type: ignore[no-untyped-def]
    return dataclasses.replace(builtin_spec("closure_sweep"), x_count=count)


def test_exact_workspace_boundaries_are_accepted() -> None:
    series = builtin_spec("swing_time_series")
    specs = (_sweep(256), _sweep(256), *(series for _ in range(6)))

    assert len(specs) == MAX_MANAGED_PLOTS
    assert plot_evaluation_count(specs) == MAX_SWEEP_EVALUATIONS
    validate_plot_workspace(specs)


def test_plot_count_and_sweep_budget_fail_closed() -> None:
    series = builtin_spec("swing_time_series")
    with pytest.raises(ValueError, match="at most 8 managed plots"):
        validate_plot_workspace((series,) * 9)
    with pytest.raises(ValueError, match="at most 512 sweep evaluations"):
        validate_plot_workspace((_sweep(257), _sweep(256)))


def test_non_sweep_plots_do_not_consume_simulation_evaluations() -> None:
    series = builtin_spec("swing_time_series")
    assert plot_evaluation_count((series,) * MAX_MANAGED_PLOTS) == 0


def test_ninth_series_is_rejected_before_workspace_computation() -> None:
    series = builtin_spec("swing_time_series")
    forged = dataclasses.replace(series, y_keys=(series.y_keys[0],) * 9)
    with pytest.raises(ValueError, match="at most 8 series"):
        validate_plot_workspace((forged,))
