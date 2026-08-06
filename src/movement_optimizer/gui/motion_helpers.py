"""Small geometry and palette helpers shared by motion-view widgets."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from PyQt6.QtGui import QColor

from movement_optimizer.rendering import Palette, get_chart_color


def build_motion_colors() -> dict[str, QColor]:
    """Return motion-canvas colors from the active fleet palette."""
    return {
        "ACCENT": QColor(Palette.GREEN),
        "CHAIN": QColor(Palette.FG_DIM),
        "BODY": QColor(get_chart_color(0)),
        "LEG": QColor(get_chart_color(1)),
        "ARM": QColor(get_chart_color(2)),
        "SURFACE": QColor(Palette.BG),
        "GRID": QColor(Palette.BG_INPUT),
    }


def chain_path_length(chain_nodes: list[tuple[float, float]]) -> float:
    """Return the polyline length with the renderer's minimum view scale."""
    distances = [
        np.hypot(end[0] - start[0], end[1] - start[1]) for start, end in pairwise(chain_nodes)
    ]
    return max(float(sum(distances)), 0.5)


__all__ = ["build_motion_colors", "chain_path_length"]
