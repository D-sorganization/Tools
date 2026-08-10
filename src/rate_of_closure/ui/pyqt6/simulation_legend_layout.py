"""Non-obscuring legend placement for the PyQt6 simulation scene."""

from __future__ import annotations

from typing import Any

__all__ = ["clear_figure_legends", "place_simulation_legend"]

_INSIDE_LOCATIONS = {
    "inside_upper_right": "upper right",
    "inside_lower_right": "lower right",
    "inside_lower_left": "lower left",
}


def clear_figure_legends(view: Any) -> None:
    """Remove retained outside legends before rebuilding one scene."""
    figure = view._figure
    for legend in tuple(figure.legends):
        legend.remove()


def place_simulation_legend(view: Any) -> None:
    """Apply the selected legend visibility and non-obscuring placement."""
    axes = view._axes
    handles, labels = axes.get_legend_handles_labels()
    if not handles or not view.legend_visible():
        axes.set_position((0.06, 0.08, 0.88, 0.82))
        return
    location = view.legend_location()
    if location == "outside_right":
        _place_outside_right(view, handles, labels)
        return
    axes.set_position((0.06, 0.08, 0.88, 0.82))
    axes.legend(
        handles,
        labels,
        loc=_INSIDE_LOCATIONS.get(location, "upper right"),
        fontsize=7,
    )


def _place_outside_right(view: Any, handles: list[Any], labels: list[str]) -> None:
    """Reserve a measured figure rail for a visible outside legend."""
    figure = view._figure
    legend = figure.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.91),
        borderaxespad=0.0,
        fontsize=7,
    )
    renderer = view._canvas.get_renderer()
    legend_bounds = legend.get_window_extent(renderer).transformed(
        figure.transFigure.inverted()
    )
    axes_right = min(0.73, max(0.42, legend_bounds.x0 - 0.02))
    view._axes.set_position((0.05, 0.08, axes_right - 0.05, 0.82))
