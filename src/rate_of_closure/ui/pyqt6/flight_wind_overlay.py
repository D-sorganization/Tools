"""Small Matplotlib helpers for selected-wind and no-wind trajectory pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover - theme package ships in-repo

    def get_chart_color(index: int) -> str:
        """Return a theme-neutral Matplotlib cycle color."""
        return f"C{index % 10}"


@dataclass(frozen=True)
class WindTrajectoryPair:
    """Selected and calm app-frame paths with palette-derived colors."""

    selected: np.ndarray
    calm: np.ndarray
    selected_color: str
    calm_color: str

    @property
    def has_comparison(self) -> bool:
        """Return whether the calm path can be rendered."""
        return len(self.calm) >= 2


def make_wind_pair(selected: np.ndarray, calm: np.ndarray) -> WindTrajectoryPair:
    """Build a trajectory pair using the canonical chart palette."""
    return WindTrajectoryPair(
        selected=selected,
        calm=calm,
        selected_color=get_chart_color(2),
        calm_color=get_chart_color(0),
    )


def plot_pair_2d(axes: object, pair: WindTrajectoryPair, ordinate: int) -> None:
    """Draw downrange against one app-frame ordinate."""
    if pair.has_comparison:
        axes.plot(  # type: ignore[attr-defined]
            pair.calm[:, 0],
            pair.calm[:, ordinate],
            color=pair.calm_color,
            lw=1.2,
            ls="--",
            label="No wind",
        )
    axes.plot(  # type: ignore[attr-defined]
        pair.selected[:, 0],
        pair.selected[:, ordinate],
        color=pair.selected_color,
        lw=1.6,
        label="Selected wind",
    )


def plot_pair_3d(axes: object, pair: WindTrajectoryPair) -> None:
    """Draw the app-frame pair using the flight view's display-axis order."""
    if pair.has_comparison:
        axes.plot(  # type: ignore[attr-defined]
            pair.calm[:, 2],
            pair.calm[:, 0],
            pair.calm[:, 1],
            color=pair.calm_color,
            lw=1.2,
            ls="--",
            label="No wind",
        )
    axes.plot(  # type: ignore[attr-defined]
        pair.selected[:, 2],
        pair.selected[:, 0],
        pair.selected[:, 1],
        color=pair.selected_color,
        lw=1.6,
        label="Selected wind",
    )


def plot_wind_pair_2d(
    axes: object, selected: np.ndarray, calm: np.ndarray, ordinate: int
) -> None:
    """Build and draw a selected/calm trajectory pair in two dimensions."""
    plot_pair_2d(axes, make_wind_pair(selected, calm), ordinate)


def plot_wind_pair_3d(axes: object, selected: np.ndarray, calm: np.ndarray) -> None:
    """Build and draw a selected/calm trajectory pair in three dimensions."""
    plot_pair_3d(axes, make_wind_pair(selected, calm))


__all__ = ["plot_wind_pair_2d", "plot_wind_pair_3d"]
