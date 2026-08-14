"""Engineering drawing markers shared by Rate of Closure 3-D views."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["draw_cg_marker"]


def draw_cg_marker(
    axes: Any, point: np.ndarray, color: str, *, label: str, abbreviation: str
) -> None:
    """Draw a high-contrast circled-cross reference symbol."""
    axes.scatter(
        *point,
        facecolors="none",
        edgecolors=color,
        s=150,
        linewidths=1.8,
        marker="o",
        zorder=7,
    )
    axes.scatter(
        *point,
        color=color,
        s=105,
        linewidths=1.8,
        marker="+",
        label=label,
        zorder=8,
    )
    axes.text(*point, f"  {abbreviation}", color=color, fontsize=8, zorder=8)
