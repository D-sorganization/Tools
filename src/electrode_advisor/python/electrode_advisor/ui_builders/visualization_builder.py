"""Visualization Builder for Electrode Advisor

Handles creation of 3D visualization tab with matplotlib widgets and charts.
"""

import logging
from collections.abc import Callable

# Matplotlib imports
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
else:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class VisualizationBuilder:
    """Builds the 3D visualization tab for the electrode advisor"""

    def __init__(self, parent_widget: Any) -> None:
        """Initialize the builder

        Args:
            parent_widget: The parent widget that will contain the visualization

        """
        self.parent = parent_widget
        self.viz_widget: QWidget | None = None

        # Matplotlib widgets references
        # Matplotlib widgets references
        self.current_fig: Figure | None = None
        self.current_canvas: FigureCanvas | None = None
        self.current_ax: Any | None = None
        self.power_fig: Figure | None = None
        self.power_canvas: FigureCanvas | None = None
        self.power_ax: Any | None = None
        self.electrode_fig: Figure | None = None
        self.electrode_canvas: FigureCanvas | None = None
        self.electrode_ax: Any | None = None

    def create_visualization_tab(
        self,
        scroll_callback: Callable,
        mouse_press_callback: Callable,
        mouse_release_callback: Callable,
        mouse_motion_callback: Callable,
    ) -> QWidget:
        """Create visualization tab with charts and 3D plot only (controls moved to separate panel)

        Args:
            scroll_callback: Callback for mouse scroll events
            mouse_press_callback: Callback for mouse press events
            mouse_release_callback: Callback for mouse release events
            mouse_motion_callback: Callback for mouse motion events

        Returns:
            QWidget: The created visualization tab

        """
        self.viz_widget = QWidget()
        viz_layout = QHBoxLayout(self.viz_widget)

        # Create matplotlib widgets with error handling
        try:
            # Charts section - stacked vertically on the left
            charts_widget = QWidget()
            charts_layout = QVBoxLayout(charts_widget)
            charts_layout.setContentsMargins(0, 0, 0, 0)
            charts_layout.setSpacing(8)

            # Current distribution chart
            self.current_fig = Figure(
                figsize=(2.5, 3.5),
            )
            self.current_fig.subplots_adjust(
                bottom=0.12,
                top=0.92,
                left=0.25,
                right=0.95,
            )
            self.current_canvas = FigureCanvas(self.current_fig)
            self.current_ax = self.current_fig.add_subplot(111)
            charts_layout.addWidget(self.current_canvas)

            # Power distribution chart
            self.power_fig = Figure(
                figsize=(2.5, 3.5),
            )
            assert self.power_fig is not None
            self.power_fig.subplots_adjust(bottom=0.12, top=0.92, left=0.25, right=0.95)
            self.power_canvas = FigureCanvas(self.power_fig)
            self.power_ax = self.power_fig.add_subplot(111)
            charts_layout.addWidget(self.power_canvas)

            # Add charts to the left of the visualization
            viz_layout.addWidget(charts_widget, 1)  # Charts get proportional space

            # 3D Electrode view - larger since controls are in separate panel
            self.electrode_fig = Figure(
                figsize=(4, 3),
            )
            self.electrode_fig.subplots_adjust(
                top=0.98,
                bottom=0.02,
                left=0.02,
                right=0.98,
            )
            self.electrode_canvas = FigureCanvas(self.electrode_fig)
            self.electrode_ax = self.electrode_fig.add_subplot(111, projection="3d")
            self.electrode_ax.set_title("")

            # Connect mouse events for 3D interaction
            self.electrode_canvas.mpl_connect("scroll_event", scroll_callback)
            self.electrode_canvas.mpl_connect(
                "button_press_event",
                mouse_press_callback,
            )
            self.electrode_canvas.mpl_connect(
                "button_release_event",
                mouse_release_callback,
            )
            self.electrode_canvas.mpl_connect(
                "motion_notify_event",
                mouse_motion_callback,
            )

            viz_layout.addWidget(self.electrode_canvas, 3)  # 3D plot gets most space

        except (RuntimeError, AttributeError) as e:
            logger.exception(
                "Error creating matplotlib widgets in VisualizationBuilder: %s", e
            )
            # Create fallback placeholder
            placeholder = QLabel("3D Visualization (Matplotlib Error)")
            placeholder.setStyleSheet(
                "background-color: #f0f0f0; border: 1px solid #ccc; padding: 20px;",
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            viz_layout.addWidget(placeholder)

        return self.viz_widget

    def get_visualization_widgets(self) -> dict[str, Any]:
        """Get all visualization widgets for external access"""
        return {
            "viz_widget": self.viz_widget,
            "current_fig": self.current_fig,
            "current_canvas": self.current_canvas,
            "current_ax": self.current_ax,
            "power_fig": self.power_fig,
            "power_canvas": self.power_canvas,
            "power_ax": self.power_ax,
            "electrode_fig": self.electrode_fig,
            "electrode_canvas": self.electrode_canvas,
            "electrode_ax": self.electrode_ax,
        }
