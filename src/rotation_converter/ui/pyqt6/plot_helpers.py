"""Shared plot helpers for rotation converter tabs.

Extracts common plotting utilities (theme colors, figure styling,
vector/matrix formatting) from the former monolithic main_window.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

# ── Theme integration (optional — graceful fallback) ──────────────
_THEME_AVAILABLE = False
try:
    from theme import (
        get_theme_manager,
        is_dark_theme,
    )

    _THEME_AVAILABLE = True
except ImportError:
    pass

# ── Default colours (used when theme system is unavailable) ───────
_DARK_BG = "#1e1e2e"
_DARK_FG = "#cdd6f4"
_DARK_ACCENT = "#89b4fa"
_DARK_SURFACE = "#313244"
_AXIS_COLORS = ["#f38ba8", "#a6e3a1", "#89b4fa"]  # RGB axes

EULER_CONVENTIONS = [
    "xyz",
    "xzy",
    "yxz",
    "yzx",
    "zxy",
    "zyx",
    "xyx",
    "xzx",
    "yxy",
    "yzy",
    "zxz",
    "zyz",
]


def fmt_vec(v: np.ndarray, decimals: int = 6) -> str:
    """Format a numpy vector as a readable string."""
    return "  ".join(f"{x: .{decimals}f}" for x in v)


def fmt_mat(M: np.ndarray, decimals: int = 6) -> str:
    """Format a numpy matrix as a multi-line string."""
    if M is None:
        raise ValueError("M must be provided")
    lines: list[str] = []
    for row in M:
        lines.append("  ".join(f"{x: .{decimals}f}" for x in row))
    return "\n".join(lines)


def parse_vec(text: str) -> np.ndarray | None:
    """Parse a whitespace/comma separated string into a numpy array."""
    try:
        parts = text.replace(",", " ").split()
        return np.array([float(p) for p in parts])
    except (ValueError, TypeError):
        return None


def get_plot_colors() -> dict[str, Any]:
    """Get current plot colours from theme or defaults."""
    if _THEME_AVAILABLE:
        try:
            from theme.colors import CHART_COLORS

            mgr = get_theme_manager()
            colors = mgr.get_current_colors()
            _dark = is_dark_theme(colors.get("name", "dark"))  # noqa: F841
            return {
                "bg": colors.get("bg", _DARK_BG),
                "fg": colors.get("text", _DARK_FG),
                "accent": colors.get("accent", _DARK_ACCENT),
                "surface": colors.get("group_bg", _DARK_SURFACE),
                "axes": CHART_COLORS[:3] if CHART_COLORS else _AXIS_COLORS,
            }
        except Exception:  # noqa: BLE001
            # Theme manager may be unavailable or return unexpected data;
            # fall through to the hardcoded dark-theme defaults below.
            pass
    return {
        "bg": _DARK_BG,
        "fg": _DARK_FG,
        "accent": _DARK_ACCENT,
        "surface": _DARK_SURFACE,
        "axes": _AXIS_COLORS,
    }


def style_figure(fig: Figure, ax: Any = None) -> None:
    """Apply current theme colours to a matplotlib figure."""
    if fig is None:
        raise ValueError("fig must be provided")
    c = get_plot_colors()
    fig.set_facecolor(c["bg"])
    if ax is not None:
        axes = [ax] if not isinstance(ax, list | np.ndarray) else list(ax)
        for a in axes:
            a.set_facecolor(c["surface"])
            a.tick_params(colors=c["fg"], labelsize=8)
            a.xaxis.label.set_color(c["fg"])
            a.yaxis.label.set_color(c["fg"])
            a.title.set_color(c["fg"])
            for spine in a.spines.values():
                spine.set_edgecolor(c["fg"])
