"""Rendering helpers for prepared geometric variability data."""

from __future__ import annotations

import numpy as np

from rate_of_closure.variation.geometric_plot_data import GeometricVariabilityData


def draw_principal_spread(axes, data: GeometricVariabilityData) -> None:  # type: ignore[no-untyped-def]
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


def draw_variability_timeline(canvas, data: GeometricVariabilityData) -> None:  # type: ignore[no-untyped-def]
    """Draw RMS positional dispersion and declared quiet samples over time."""
    axes = canvas.axes
    axes.clear()
    canvas.apply_theme()
    times = data.sample_times_s
    rms_mm = data.rms_radius_m * 1000.0
    axes.plot(times, rms_mm, color="#38bdf8", linewidth=1.6, label="RMS Radius")
    axes.fill_between(
        times,
        0.0,
        rms_mm,
        where=data.quiet_mask,
        color="#34d399",
        alpha=0.28,
        label="Quiet Zone",
    )
    axes.axhline(
        data.criteria.max_rms_radius_m * 1000.0,
        color="#fbbf24",
        linestyle="--",
        linewidth=1.0,
        label="Quiet Threshold",
    )
    axes.set_xlabel("Common Simulation Time [s]")
    axes.set_ylabel("RMS Position Radius [mm]")
    axes.set_title("Geometric Variability and Quiet Zones")
    axes.legend(loc="best", fontsize=8)
    canvas.draw_idle()


__all__ = ["draw_principal_spread", "draw_variability_timeline"]
