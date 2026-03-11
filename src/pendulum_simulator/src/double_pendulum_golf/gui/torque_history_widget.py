"""
Torque history plot widget.

Displays the full time history of driving, friction, and total applied torques
at each joint after the simulation completes. This provides a post-simulation
analysis view showing how the frictional dissipation compares to the user-driven
torque at each instant.

Theme integration
-----------------
Background, axis, and text colours are sourced from the shared PlotThemeManager
when available, falling back to hardcoded dark defaults.  The distinctive *trace*
colours (warm orange, cool blue, red, teal, gold, pale green) are preserved
regardless of theme — these are the signature look of the plots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from ..simulation import SimulationResult

logger = logging.getLogger(__name__)

try:
    import pyqtgraph as pg

    _HAS_PYQTGRAPH = True
except ImportError:
    _HAS_PYQTGRAPH = False

# ── Try to import shared PlotThemeManager ──────────────────────────────────
_PLOT_THEME_AVAILABLE = False
_get_plot_theme_manager: Any = None
try:
    import sys
    from pathlib import Path

    _shared_root = None
    _p = Path(__file__).resolve().parent
    for _ in range(10):
        _candidate = _p / "shared" / "python"
        if _candidate.exists():
            _shared_root = _candidate
            break
        _p = _p.parent
    if _shared_root is not None and str(_shared_root) not in sys.path:
        sys.path.insert(0, str(_shared_root))

    from plot_theme.manager import (
        get_plot_theme_manager as _shared_get_plot_theme_manager,
    )

    _get_plot_theme_manager = _shared_get_plot_theme_manager
    _PLOT_THEME_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Trace colour scheme — PRESERVED across all themes (user's favourite part)
# ---------------------------------------------------------------------------
_COLOR_DRIVE_1 = (230, 120, 50)  # warm orange — shoulder drive
_COLOR_DRIVE_2 = (120, 180, 230)  # cool blue   — wrist drive
_COLOR_FRICTION_1 = (200, 80, 80)  # red         — shoulder friction
_COLOR_FRICTION_2 = (80, 160, 160)  # teal        — wrist friction
_COLOR_TOTAL_1 = (255, 220, 80)  # gold        — shoulder total
_COLOR_TOTAL_2 = (180, 255, 180)  # pale green  — wrist total

# Default dark background/text when no theme system is available
_DEFAULT_BG = "#1a1a28"
_DEFAULT_TEXT = "#c0c0d8"
_DEFAULT_GRID = "#303050"


class TorqueHistoryWidget(QWidget):
    """Post-simulation torque breakdown chart.

    Shows three torque series per joint over the full simulation time:
        * Driving torque   — from the user-specified polynomial torque function
        * Friction torque  — viscous damping + Coulomb friction (always ≤ 0 N·m
                             in magnitude when resisting motion)
        * Total torque     — driving + friction (net torque entering EOM)

    When pyqtgraph is available the chart is rendered with hardware-accelerated
    line plots. If pyqtgraph is missing a plain text fallback is shown.

    Contract:
        - set_simulation() must be called before set_frame() or clear().
        - clear() resets to an empty state without crashing.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: SimulationResult | None = None
        self._bg_color = _DEFAULT_BG
        self._text_color = _DEFAULT_TEXT
        self._grid_color = _DEFAULT_GRID
        self._load_theme_colors()
        self._build_ui()

    # ------------------------------------------------------------------
    # Theme integration
    # ------------------------------------------------------------------

    def _load_theme_colors(self) -> None:
        """Load background/text/grid from PlotThemeManager if available."""
        if not _PLOT_THEME_AVAILABLE or _get_plot_theme_manager is None:
            return
        try:
            manager = _get_plot_theme_manager(settings_app="PendulumSimulator")
            theme = manager.current_theme
            self._bg_color = theme.axes_facecolor
            self._text_color = theme.text_color
            self._grid_color = theme.grid_color
            # Register for future theme changes
            manager.add_theme_change_callback(self._on_plot_theme_changed)
        except Exception:
            logger.debug("PlotThemeManager unavailable, using defaults")

    def _on_plot_theme_changed(self, theme: object) -> None:
        """Update backgrounds when the plot theme changes (trace colors stay)."""
        if not _HAS_PYQTGRAPH:
            return
        try:
            self._bg_color = theme.axes_facecolor  # type: ignore[attr-defined]
            self._text_color = theme.text_color  # type: ignore[attr-defined]
            self._grid_color = theme.grid_color  # type: ignore[attr-defined]
            for pw in (self._plot_j1, self._plot_j2):
                pw.setBackground(self._bg_color)
        except Exception:
            logger.debug("Could not update torque plot theme")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("📊 Torque History")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {self._text_color}; font-size: 14px; font-weight: bold;"
            "padding: 4px; border-bottom: 1px solid #505070;"
        )
        layout.addWidget(title)

        if not _HAS_PYQTGRAPH:
            fallback = QLabel(
                "Install pyqtgraph for torque plots:\n  pip install pyqtgraph"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color: #808090; font-size: 11px;")
            layout.addWidget(fallback)
            return

        # Style for clearer axis text (#1145)
        _axis_style = {"color": self._text_color, "font-size": "11px"}

        # Two stacked plot widgets (joint 1 top, joint 2 bottom)
        self._plot_j1 = pg.PlotWidget(title="Joint 1 — Shoulder")
        self._plot_j2 = pg.PlotWidget(title="Joint 2 — Wrist")

        for pw in (self._plot_j1, self._plot_j2):
            pw.setBackground(self._bg_color)
            pw.getPlotItem().setLabel("bottom", "Time (s)", **_axis_style)
            pw.getPlotItem().setLabel("left", "Torque (N·m)", **_axis_style)
            pw.getPlotItem().addLegend(offset=(10, 10))
            pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(pw)

        # Pre-create curve objects (hidden until data loaded)
        pen_kwargs = {"width": 2}
        self._curves_j1: dict[str, pg.PlotDataItem] = {
            "drive": self._plot_j1.plot(
                name="Drive", pen=pg.mkPen(*_COLOR_DRIVE_1, **pen_kwargs)
            ),
            "friction": self._plot_j1.plot(
                name="Friction",
                pen=pg.mkPen(
                    *_COLOR_FRICTION_1, style=Qt.PenStyle.DashLine, **pen_kwargs
                ),
            ),
            "total": self._plot_j1.plot(
                name="Total",
                pen=pg.mkPen(*_COLOR_TOTAL_1, style=Qt.PenStyle.DotLine, **pen_kwargs),
            ),
        }
        self._curves_j2: dict[str, pg.PlotDataItem] = {
            "drive": self._plot_j2.plot(
                name="Drive", pen=pg.mkPen(*_COLOR_DRIVE_2, **pen_kwargs)
            ),
            "friction": self._plot_j2.plot(
                name="Friction",
                pen=pg.mkPen(
                    *_COLOR_FRICTION_2, style=Qt.PenStyle.DashLine, **pen_kwargs
                ),
            ),
            "total": self._plot_j2.plot(
                name="Total",
                pen=pg.mkPen(*_COLOR_TOTAL_2, style=Qt.PenStyle.DotLine, **pen_kwargs),
            ),
        }

        # Vertical time cursor (updates with animation frame)
        self._cursor_j1 = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine),
        )
        self._cursor_j2 = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot_j1.addItem(self._cursor_j1)
        self._plot_j2.addItem(self._cursor_j2)

    # ------------------------------------------------------------------
    # Public interface (mirrors matrix_widget.py contract)
    # ------------------------------------------------------------------

    def set_simulation(self, result: SimulationResult) -> None:
        """Load simulation result and compute full torque history arrays.

        All N timesteps' torques are computed once here so that set_frame()
        can do only an O(1) cursor update during animation.

        Preconditions:
            - result is a completed SimulationResult with n_steps >= 2.
        """
        assert result.n_steps >= 2, "Result must have at least 2 time steps"
        self._result = result

        if not _HAS_PYQTGRAPH:
            return

        t = result.t
        n = result.n_steps

        # Pre-allocate arrays for all torque components
        drive = np.empty((n, 2))
        friction = np.empty((n, 2))
        total = np.empty((n, 2))

        for i in range(n):
            drive[i] = result.torques_at(i)
            friction[i] = result.friction_torques_at(i)
            total[i] = result.total_torques_at(i)

        # Joint 1 (shoulder)
        self._curves_j1["drive"].setData(t, drive[:, 0])
        self._curves_j1["friction"].setData(t, friction[:, 0])
        self._curves_j1["total"].setData(t, total[:, 0])

        # Joint 2 (wrist)
        self._curves_j2["drive"].setData(t, drive[:, 1])
        self._curves_j2["friction"].setData(t, friction[:, 1])
        self._curves_j2["total"].setData(t, total[:, 1])

        # Reset cursors to t=0
        self._cursor_j1.setValue(t[0])
        self._cursor_j2.setValue(t[0])

    def set_frame(self, idx: int) -> None:
        """Move the time cursor to the frame at index idx.

        Called every animation tick — must be O(1).

        Preconditions:
            - set_simulation() has been called.
            - 0 <= idx < result.n_steps.
        """
        if self._result is None or not _HAS_PYQTGRAPH:
            return
        assert 0 <= idx < self._result.n_steps
        t_now = self._result.t[idx]
        self._cursor_j1.setValue(t_now)
        self._cursor_j2.setValue(t_now)

    def clear(self) -> None:
        """Reset to an empty state (called on simulation reset)."""
        self._result = None
        if not _HAS_PYQTGRAPH:
            return
        for curve in (*self._curves_j1.values(), *self._curves_j2.values()):
            curve.setData([], [])
        self._cursor_j1.setValue(0.0)
        self._cursor_j2.setValue(0.0)
