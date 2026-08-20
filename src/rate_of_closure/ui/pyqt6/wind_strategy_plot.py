"""Managed wind-strategy scatter rendering and availability text."""

from __future__ import annotations

from matplotlib.figure import Figure

from rate_of_closure.variation.scalar_ensemble_contract import ScalarScatterData

_COHORT_COLORS = {
    "completed": "#2f8bd6",
    "nonconverged": "#eb9f3c",
    "invalid": "#d35f5f",
}


def draw_wind_strategy_scatter(
    figure: Figure,
    scatter: ScalarScatterData,
) -> None:
    """Draw every paired-finite cohort into a managed Matplotlib figure."""
    figure.clear()
    axes = figure.add_subplot(111)
    for cohort, color in _COHORT_COLORS.items():
        points = [point for point in scatter.points if point.cohort == cohort]
        if points:
            axes.scatter(
                [point.x for point in points],
                [point.y for point in points],
                s=22,
                alpha=0.75,
                label=cohort.title(),
                color=color,
            )
    axes.set_xlabel(f"{scatter.x_variable.label} [{scatter.x_variable.unit}]")
    axes.set_ylabel(f"{scatter.y_variable.label} [{scatter.y_variable.unit}]")
    axes.set_title("Wind Strategy Trial Scatter")
    axes.grid(alpha=0.25)
    if axes.collections:
        axes.legend(loc="best", fontsize=8)


def scatter_availability_text(scatter: ScalarScatterData) -> str:
    """Format exact overall and per-cohort paired-finite counts."""
    available = scatter.availability.overall
    cohorts = ", ".join(
        f"{key}: {item.paired_finite}/{item.total_rows}"
        for key, item in scatter.availability.by_cohort.items()
    )
    return (
        f"Paired finite {available.paired_finite}/{available.total_rows}; "
        f"unavailable {available.unavailable}. {cohorts}"
    )


__all__ = ["draw_wind_strategy_scatter", "scatter_availability_text"]
