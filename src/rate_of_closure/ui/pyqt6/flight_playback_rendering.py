"""Mutable Matplotlib ball markers for the otherwise static flight plots."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover - theme package ships in-repo

    def get_chart_color(index: int) -> str:
        """Matplotlib cycle colors as a theme-neutral fallback."""
        return f"C{index % 10}"


class FlightPlaybackArtists:
    """Own and update moving markers without rebuilding axes or camera state."""

    def __init__(self) -> None:
        self._position_m: np.ndarray | None = None
        self._artists_2d: list[tuple[Any, int]] = []
        self._artists_3d: list[Any] = []

    def reset(self, position_m: np.ndarray | None) -> None:
        """Forget artists after a full figure redraw and adopt a position."""
        self._position_m = position_m
        self._artists_2d.clear()
        self._artists_3d.clear()

    def add_2d(self, axes: Any, vertical_index: int) -> None:
        """Add the moving ball to a carry-versus-component panel."""
        if self._position_m is None:
            return
        point = self._position_m
        artist = axes.scatter(
            [point[0]],
            [point[vertical_index]],
            s=60,
            color=get_chart_color(1),
            edgecolors="white",
            linewidths=0.8,
            zorder=8,
            label="Playback ball",
        )
        self._artists_2d.append((artist, vertical_index))

    def add_3d(self, axes: Any) -> None:
        """Add the moving ball to the app-frame 3D panel."""
        if self._position_m is None:
            return
        point = self._position_m
        artist = axes.scatter(
            [point[2]], [point[0]], [point[1]], s=70, color=get_chart_color(1)
        )
        self._artists_3d.append(artist)

    def update(self, position_m: np.ndarray) -> None:
        """Move all live markers while preserving rotation and zoom state."""
        self._position_m = position_m
        for artist, vertical_index in self._artists_2d:
            artist.set_offsets([[position_m[0], position_m[vertical_index]]])
        for artist in self._artists_3d:
            artist._offsets3d = ([position_m[2]], [position_m[0]], [position_m[1]])


__all__ = ["FlightPlaybackArtists"]
