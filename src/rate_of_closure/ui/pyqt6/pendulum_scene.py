"""Articulated pendulum drawing for the PyQt6 swing scene."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["draw_pendulum_skeleton"]


def draw_pendulum_skeleton(
    axes: Any,
    joints: np.ndarray,
    chart_color: Callable[[int], str],
) -> None:
    """Draw sampled joints and links already transformed to display axes."""
    for link_index in range(len(joints) - 1):
        segment = joints[link_index : link_index + 2].T
        is_club = link_index == len(joints) - 2
        axes.plot(
            *segment,
            color=chart_color(7 if is_club else 4),
            lw=4.0 if is_club else 6.0,
            solid_capstyle="round",
            label=(
                "club link"
                if is_club
                else ("pendulum skeleton" if link_index == 0 else None)
            ),
            zorder=6,
        )
    axes.scatter(
        joints[:, 0],
        joints[:, 1],
        joints[:, 2],
        color=chart_color(6),
        edgecolors="white",
        linewidths=0.7,
        s=34,
        zorder=7,
    )
