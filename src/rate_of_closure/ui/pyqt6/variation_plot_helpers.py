"""Presentation helpers shared by PyQt variation visualization widgets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from rate_of_closure.variation.plot_data import (
    ArcOverlayData,
    CohortAvailability,
    ScalarPlotVariable,
)
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.swing_sim.variation import VariationDataset


def axis_label(variable: ScalarPlotVariable) -> str:
    """Return a compact, unit-bearing plot label."""
    return f"{variable.label} [{variable.unit}]" if variable.unit else variable.label


def cohort_label(cohort: TrialEvaluationStatus) -> str:
    """Return the concise UI label for a typed trial cohort."""
    return {
        TrialEvaluationStatus.EVALUATED_HIT: "Hit",
        TrialEvaluationStatus.EVALUATED_NO_IMPACT: "No Impact",
        TrialEvaluationStatus.NUMERICAL_FAILURE: "Numerical Failure",
    }[cohort]


def availability_text(summaries: Iterable[CohortAvailability]) -> str:
    """Describe plotted/unavailable counts without selection bias."""
    return " · ".join(
        f"{cohort_label(summary.cohort)}: {summary.plotted}/{summary.total} plotted"
        + (f", {summary.unavailable} unavailable" if summary.unavailable else "")
        for summary in summaries
    )


def point_label(point_id: str) -> str:
    """Convert a stable spatial point ID into a title-case label."""
    return point_id.rsplit(".", 1)[-1].replace("_", " ").title()


def dataset_values(
    dataset: VariationDataset,
    variable: ScalarPlotVariable,
) -> np.ndarray:
    """Return one all-row scalar column without silently dropping failures."""
    source, name = variable.key.split(":", 1)
    if source == "input":
        return np.asarray(dataset.inputs[:, dataset.input_names.index(name)])
    return np.asarray(dataset.outputs[:, dataset.output_names.index(name)])


def equal_3d_axes(axes: Any, overlay: ArcOverlayData) -> None:
    """Set one physical scale across all three spatial axes."""
    finite = overlay.positions_m[np.isfinite(overlay.positions_m).all(axis=-1)]
    if finite.size == 0:
        return
    plot_xyz = finite[:, [0, 2, 1]]
    low = np.min(plot_xyz, axis=0)
    high = np.max(plot_xyz, axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1e-6)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    axes.set_box_aspect((1.0, 1.0, 1.0))


__all__ = [
    "availability_text",
    "axis_label",
    "cohort_label",
    "dataset_values",
    "equal_3d_axes",
    "point_label",
]
