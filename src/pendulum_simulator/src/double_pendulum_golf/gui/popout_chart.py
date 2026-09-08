"""
Pop-out chart window with regression fitting (#1135).

Provides a detachable matplotlib window that displays simulation data
(torque history, energy, etc.) with optional polynomial regression
overlay.  The window is non-modal so users can keep it open alongside
the animation.

Design by Contract
------------------
- ``PopOutChart.plot_data()`` requires x and y arrays of equal length.
- Regression degree must be >= 0 and <= 10.
- If matplotlib is unavailable, a graceful fallback message is shown.

DRY
---
Regression logic is isolated in ``fit_regression()`` for reuse.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except (ImportError, RuntimeError):
    Figure = None  # type: ignore[assignment, misc]
    FigureCanvasQTAgg = None  # type: ignore[assignment, misc]
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Regression utility
# ---------------------------------------------------------------------------


def fit_regression(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a polynomial regression and return evaluation points.

    Parameters
    ----------
    x, y : ndarray, shape (n,)
        Data points.
    degree : int
        Polynomial degree (0–10).

    Returns
    -------
    x_fit : ndarray
        Dense x grid for smooth curve.
    y_fit : ndarray
        Fitted y values on the dense grid.
    coeffs : ndarray
        Polynomial coefficients [highest → lowest].

    Pre: len(x) == len(y), degree >= 0, degree <= 10.
    Post: x_fit and y_fit have same length; len(coeffs) == degree + 1.
    """
    if not (len(x) == len(y)):
        raise ValueError(f"x and y must have same length: {len(x)} vs {len(y)}")
    if not (0 <= degree <= 10):
        raise ValueError(f"Degree must be 0–10, got {degree}")

    coeffs = np.polyfit(x, y, degree)
    x_fit = np.linspace(x.min(), x.max(), max(200, len(x)))
    y_fit = np.polyval(coeffs, x_fit)
    return x_fit, y_fit, coeffs


# ---------------------------------------------------------------------------
# Pop-out chart widget
# ---------------------------------------------------------------------------


class PopOutChart:
    """Non-modal pop-out chart window with regression capability.

    Usage::

        chart = PopOutChart(parent)
        chart.plot_data(t, values, "Time (s)", "Torque (N·m)", "Joint 1 Torque")
        chart.add_regression(degree=3)
        chart.show()
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        self._parent = parent
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._xlabel = ""
        self._ylabel = ""
        self._title = ""
        self._fig: Figure | None = None
        self._ax: object | None = None
        self._canvas: object | None = None
        self._window: object | None = None
        self._regression: tuple[np.ndarray, np.ndarray, int, np.ndarray] | None = None
        self._regression_label: str = ""

    def plot_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = "X",
        ylabel: str = "Y",
        title: str = "Chart",
    ) -> None:
        """Store data for plotting.

        Pre: x and y have same length, both finite.
        """
        if not (len(x) == len(y)):
            raise ValueError("x and y must have same length")
        self._x = np.asarray(x)
        self._y = np.asarray(y)
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._title = title

    def add_regression(self, degree: int = 2) -> tuple[np.ndarray, np.ndarray] | None:
        """Fit and store a regression line.

        Returns (x_fit, y_fit) or None if no data.
        """
        if degree is None:
            raise ValueError("degree must be provided")
        if self._x is None or self._y is None:
            return None
        x_fit, y_fit, coeffs = fit_regression(self._x, self._y, degree)
        self._regression = (x_fit, y_fit, degree, coeffs)
        # Format coefficients for display
        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if abs(c) < 1e-12:
                continue
            terms.append(f"{c:+.4g}t^{power}" if power > 0 else f"{c:+.4g}")
        self._regression_label = " ".join(terms) if terms else "0"
        logger.info(
            "Regression (degree %d): %s",
            degree,
            self._regression_label,
        )
        return x_fit, y_fit

    def show(self) -> None:
        """Display the pop-out chart window."""
        if not _HAS_MPL:
            logger.warning("matplotlib not available — cannot show pop-out chart")
            try:
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.information(
                    self._parent,
                    "Pop-Out Chart",
                    "Install matplotlib for pop-out charts:\n  pip install matplotlib",
                )
            except (ImportError, RuntimeError) as exc:
                logger.debug("Could not show matplotlib info dialog: %s", exc)
            return

        if self._x is None or self._y is None:
            logger.warning("No data to plot")
            return

        # Create figure
        fig_cls = Figure
        if fig_cls is None:
            from matplotlib.figure import Figure as fig_cls  # type: ignore[no-redef]
        self._fig = fig_cls(figsize=(8, 5), dpi=100)
        # ``add_subplot`` is untyped (matplotlib); bind the local directly so the
        # plotting calls below are Any-typed rather than narrowed to ``object``
        # via the ``self._ax: object | None`` attribute annotation.
        ax: Any = self._fig.add_subplot(111)
        self._ax = ax

        from shared.python.theme.integration import get_theme_manager
        from shared.python.theme.matplotlib_style import apply_plot_theme

        apply_plot_theme(self._fig, get_theme_manager().get_current_colors())

        # Plot data
        ax.plot(self._x, self._y, color="#6fa8dc", linewidth=1.5, label="Data")

        # Plot regression if available
        if self._regression is not None:
            x_fit, y_fit, degree, _ = self._regression
            ax.plot(
                x_fit,
                y_fit,
                color="#ff7043",
                linewidth=2,
                linestyle="--",
                label=f"Fit (deg {degree})",
            )

        ax.set_xlabel(self._xlabel)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.legend()
        ax.grid(True, alpha=0.5)

        self._fig.tight_layout()

        # Show in a separate window
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

        from shared.python.theme.integration import ThemedWindowMixin

        class PopOutWindow(ThemedWindowMixin, QMainWindow):
            pass

        self._window = PopOutWindow(self._parent)
        self._window.setup_theme_support()
        self._window.setWindowTitle(self._title)
        self._window.setMinimumSize(700, 450)
        self._window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        canvas = FigureCanvasQTAgg(self._fig)
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(canvas)
        self._window.setCentralWidget(central)
        self._window.show()

        logger.info("Opened pop-out chart: %s", self._title)
