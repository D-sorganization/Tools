"""Rendering helpers for prepared geometric variability data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rate_of_closure.variation.confidence_ellipsoid_mesh import (
    ConfidenceEllipsoidMesh,
)
from rate_of_closure.variation.geometric_plot_data import (
    DispersionMetricVariabilityData,
)
from rate_of_closure.variation.plot_data import ArcOverlayData
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus
from shared.python.swing_sim.variation import (
    ELLIPSOID_VOLUME,
    LARGEST_PRINCIPAL_SIGMA,
    RMS_RADIUS,
)

_METRIC_LABELS = {
    RMS_RADIUS: "RMS Position Radius",
    LARGEST_PRINCIPAL_SIGMA: "Largest Principal Sigma",
    ELLIPSOID_VOLUME: "Confidence Ellipsoid Volume",
}


def draw_arc_trials(
    axes: Any,
    overlay: ArcOverlayData,
    selected_trial: int | None,
    cohort_colors: Mapping[TrialEvaluationStatus, str],
) -> None:
    """Draw valid cohort-colored arcs and their shared median reference."""
    for trial_index, positions, valid, cohort in zip(
        overlay.trial_indices,
        overlay.positions_m,
        overlay.sample_valid,
        overlay.cohorts,
        strict=True,
    ):
        if not np.any(valid):
            continue
        selected = trial_index == selected_trial
        axes.plot(
            positions[valid, 0],
            positions[valid, 2],
            positions[valid, 1],
            color=cohort_colors[cohort],
            linewidth=2.8 if selected else 0.8,
            alpha=1.0 if selected else (0.12 if selected_trial is not None else 0.34),
        )
    reference = overlay.reference_positions_m
    axes.plot(
        reference[:, 0],
        reference[:, 2],
        reference[:, 1],
        color="#f2f4f8",
        linewidth=2.2,
        label="Median Reference",
    )


def draw_principal_spread(axes: Any, data: DispersionMetricVariabilityData) -> None:
    """Draw sparse two-sigma principal-axis glyphs in the app frame."""
    count = data.sample_times_s.size
    stride = max(1, count // 14)
    for index in range(0, count, stride):
        sigma = data.principal_sigma_m[index, 0]
        axis = data.principal_axes[index, :, 0]
        mean = data.mean_positions_m[index]
        if not np.isfinite(sigma) or not np.all(np.isfinite(axis + mean)):
            continue
        endpoints = np.stack((mean - 2.0 * sigma * axis, mean + 2.0 * sigma * axis))
        axes.plot(
            endpoints[:, 0],
            endpoints[:, 2],
            endpoints[:, 1],
            color="#fbbf24",
            linewidth=1.1,
            alpha=0.8,
        )


def draw_confidence_ellipsoid_mesh(axes: Any, mesh: ConfidenceEllipsoidMesh) -> None:
    """Draw bounded Gaussian position-content surfaces in Matplotlib coordinates."""
    if not mesh.triangles.size:
        return
    app_triangles = mesh.vertices_m[mesh.triangles]
    display_triangles = app_triangles[:, :, (0, 2, 1)]
    collection = Poly3DCollection(
        display_triangles,
        facecolor="#22d3ee",
        edgecolor="#67e8f9",
        linewidth=0.25,
        alpha=0.16,
        zsort="average",
    )
    collection.set_label("_nolegend_")
    axes.add_collection3d(collection)


def confidence_ellipsoid_legend(confidence_level: float) -> Patch:
    """Return an accessible legend proxy for translucent 3-D surfaces."""
    return Patch(
        facecolor="#22d3ee",
        edgecolor="#67e8f9",
        alpha=0.35,
        label=(
            f"{100.0 * confidence_level:.1f}% Gaussian position-content ellipsoid "
            "(not mean CI)"
        ),
    )


def principal_spread_legend() -> Line2D:
    """Return the explicit legend proxy for sparse two-sigma glyphs."""
    return Line2D(
        [0],
        [0],
        color="#fbbf24",
        linewidth=1.1,
        label="Sparse 2σ largest-principal-axis glyph",
    )


def draw_variability_timeline(
    canvas: Any, data: DispersionMetricVariabilityData
) -> None:
    """Draw the selected display-unit metric and ranked quiet samples."""
    axes = canvas.axes
    axes.clear()
    canvas.apply_theme()
    times = data.sample_times_s
    values = data.display_values
    threshold = data.criteria.max_value * (
        1_000_000_000.0 if data.authority_unit == "m^3" else 1_000.0
    )
    label = _METRIC_LABELS[data.metric]
    axes.plot(times, values, color="#38bdf8", linewidth=1.6, label=label)
    axes.fill_between(
        times,
        0.0,
        values,
        where=data.quiet_mask,
        color="#34d399",
        alpha=0.28,
        label="Quiet Zone",
    )
    axes.axhline(
        threshold,
        color="#fbbf24",
        linestyle="--",
        linewidth=1.0,
        label="Quiet Threshold",
    )
    axes.set_xlabel("Common Simulation Time [s]")
    axes.set_ylabel(f"{label} [{data.display_unit}]")
    axes.set_title("Selected Dispersion Metric and Ranked Quiet Zones")
    axes.legend(loc="best", fontsize=8)
    canvas.draw_idle()


def set_app_frame_axes(axes) -> None:  # type: ignore[no-untyped-def]
    """Label axes while plotting app y-up as visual z."""
    axes.set_xlabel("Target, x [m]")
    axes.set_ylabel("Right, z [m]")
    axes.set_zlabel("Up, y [m]")


def clear_arc_views(main_canvas, variability_canvas) -> None:  # type: ignore[no-untyped-def]
    """Reset both geometric canvases before an ensemble is available."""
    main_canvas.axes.clear()
    main_canvas.apply_theme()
    main_canvas.axes.set_title("All-Trial Swing Arc Overlay")
    set_app_frame_axes(main_canvas.axes)
    main_canvas.draw_idle()
    variability_canvas.axes.clear()
    variability_canvas.apply_theme()
    variability_canvas.axes.set_title("Geometric Variability and Quiet Zones")
    variability_canvas.draw_idle()


__all__ = [
    "clear_arc_views",
    "draw_arc_trials",
    "draw_confidence_ellipsoid_mesh",
    "draw_principal_spread",
    "draw_variability_timeline",
    "set_app_frame_axes",
    "confidence_ellipsoid_legend",
    "principal_spread_legend",
]
