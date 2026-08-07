"""Compact ranked visualization for exploratory covariation scans."""

from __future__ import annotations

from typing import Any

import pandas as pd

TOP_PAIR_COUNT = 15


def plot_covariation_scan(plot_widget: Any, ranking: pd.DataFrame) -> None:
    """Plot the highest-ranked random-effects correlations."""

    plot_widget.figure.clear()
    axis = plot_widget.figure.add_subplot(111)
    top = ranking.head(TOP_PAIR_COUNT).iloc[::-1]
    labels = top["x_column"].astype(str) + " × " + top["y_column"].astype(str)
    correlations = top["random_effect_r"]
    colors = ["#168aad" if value >= 0 else "#d62828" for value in correlations]
    axis.barh(labels, top["random_effect_r"].fillna(0.0), color=colors, alpha=0.8)
    axis.axvline(0.0, color="black", linestyle="--")
    axis.set_xlabel("Random-Effects Pearson Correlation (unitless)")
    axis.set_ylabel("Variable Pair")
    axis.set_title(f"Top {len(top)} Exploratory Within-Player Pairs")
    axis.set_xlim(-1.05, 1.05)
    axis.grid(axis="x", alpha=0.2)
    plot_widget.backing_data = ranking.copy()
    plot_widget.description = (
        "Complete deterministic ranking; displayed bars are limited to the top "
        f"{TOP_PAIR_COUNT}. Exploratory correlation does not imply causation."
    )
    plot_widget.canvas.draw_idle()


__all__ = ["plot_covariation_scan"]
