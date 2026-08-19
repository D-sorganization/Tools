"""Matplotlib geometry for a representative tee aligned to a ball setup."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d.axes3d import Axes3D

from shared.python.swing_sim.ball_setup import BallSetup, BallSupportMode
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M

__all__ = ["draw_representative_tee"]

_ANGLE_SAMPLES = 24


def draw_representative_tee(
    axes: Axes3D,
    setup: BallSetup,
    display: Callable[[np.ndarray], np.ndarray],
    color: str,
) -> tuple[Artist, ...]:
    """Draw a tapered stem and shallow cup at the configured support height."""
    if setup.support_mode is BallSupportMode.GROUND:
        return ()
    height_m = setup.tee_height_m
    if height_m <= 0.0:
        return (_draw_flat_support(axes, setup, display, color),)
    stem = _surface(
        axes,
        display,
        np.linspace(0.0, height_m * 0.82, 4),
        np.linspace(0.004, 0.0025, 4),
        color,
    )
    cup = _surface(
        axes,
        display,
        np.linspace(height_m * 0.82, height_m, 4),
        np.linspace(0.0025, GOLF_BALL_RADIUS_M * 0.55, 4),
        color,
    )
    for artist in (stem, cup):
        artist.set_gid("ball-setup-tee")
    return stem, cup


def _surface(
    axes: Axes3D,
    display: Callable[[np.ndarray], np.ndarray],
    heights_m: np.ndarray,
    radii_m: np.ndarray,
    color: str,
) -> Artist:
    theta = np.linspace(0.0, 2.0 * np.pi, _ANGLE_SAMPLES)
    heights = heights_m[:, None]
    radii = radii_m[:, None]
    app_points = np.stack(
        [
            radii * np.cos(theta),
            np.broadcast_to(heights, (len(heights_m), len(theta))),
            radii * np.sin(theta),
        ],
        axis=-1,
    )
    points = display(app_points)
    return cast(
        Artist,
        axes.plot_surface(
            points[..., 0],
            points[..., 1],
            points[..., 2],
            color=color,
            alpha=0.92,
            linewidth=0.15,
            shade=True,
        ),
    )


def _draw_flat_support(
    axes: Axes3D,
    setup: BallSetup,
    display: Callable[[np.ndarray], np.ndarray],
    color: str,
) -> Artist:
    theta = np.linspace(0.0, 2.0 * np.pi, _ANGLE_SAMPLES)
    radius = GOLF_BALL_RADIUS_M * 0.55
    center = np.asarray(setup.ball_center_m)
    points = np.column_stack(
        [
            center[0] + radius * np.cos(theta),
            np.zeros_like(theta),
            center[2] + radius * np.sin(theta),
        ]
    )
    rendered = display(points)
    (artist,) = axes.plot(*rendered.T, color=color, linewidth=2.0)
    artist.set_gid("ball-setup-tee")
    return cast(Artist, artist)
