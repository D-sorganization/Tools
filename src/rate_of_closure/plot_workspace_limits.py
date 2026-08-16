"""Bounded resource contract for the managed Plots workspace."""

from __future__ import annotations

from collections.abc import Sequence

from rate_of_closure._contracts import require
from rate_of_closure.plot_point_inspector import MAX_PLOT_SERIES
from rate_of_closure.plotting import PlotSpec

MAX_MANAGED_PLOTS = 8
MAX_SWEEP_EVALUATIONS = 512


def plot_evaluation_count(specs: Sequence[PlotSpec]) -> int:
    """Return the number of full simulation runs required by sweep plots."""
    total = 0
    for spec in specs:
        require(isinstance(spec, PlotSpec), "plot must be a PlotSpec", spec)
        require(
            len(spec.y_keys) <= MAX_PLOT_SERIES,
            f"plot supports at most {MAX_PLOT_SERIES} series",
            len(spec.y_keys),
        )
        if spec.kind == "sweep":
            total += spec.x_count
    return total


def validate_plot_workspace(specs: Sequence[PlotSpec]) -> None:
    """Reject a managed workspace above either deterministic resource cap."""
    require(
        len(specs) <= MAX_MANAGED_PLOTS,
        f"workspace supports at most {MAX_MANAGED_PLOTS} managed plots",
        len(specs),
    )
    evaluations = plot_evaluation_count(specs)
    require(
        evaluations <= MAX_SWEEP_EVALUATIONS,
        f"workspace supports at most {MAX_SWEEP_EVALUATIONS} sweep evaluations",
        evaluations,
    )


__all__ = [
    "MAX_MANAGED_PLOTS",
    "MAX_SWEEP_EVALUATIONS",
    "plot_evaluation_count",
    "validate_plot_workspace",
]
