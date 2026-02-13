"""TabsMixin -- tab creation, checkbox signals, and tab styling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
else:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from matplotlib.figure import Figure

from ...ui_builders.visualization_builder import VisualizationBuilder

logger = logging.getLogger(__name__)


class TabsMixin:
    """Mixin providing tab creation and checkbox signal wiring."""

    # -- Attributes provided by the host class (declared for mypy) --
    results_tabs: Any
    _calculate_system: Any
    _on_input_changed: Any
    _on_mouse_motion: Any
    _on_mouse_press: Any
    _on_mouse_release: Any
    _on_scroll: Any

    def _connect_checkbox_signals(self) -> None:
        """Connect all checkbox signals to ensure proper updates"""
        checkboxes = [
            "show_refractory_checkbox",
            "show_glass_checkbox",
            "show_electrodes_checkbox",
            "show_metal_shell_checkbox",
            "show_metal_checkbox",
            "show_paths_checkbox",
            "show_axis_labels_checkbox",
            "show_electrode_labels_checkbox",
            "show_current_values_checkbox",
        ]

        for checkbox_name in checkboxes:
            if hasattr(self, checkbox_name):
                checkbox = getattr(self, checkbox_name)
                # Disconnect any existing connections to avoid duplicates
                try:
                    checkbox.stateChanged.disconnect()
                except (TypeError, RuntimeError) as disconnect_error:
                    logger.debug(
                        "Checkbox '%s' disconnect skipped: %s",
                        checkbox_name,
                        disconnect_error,
                    )
                # Connect to the input changed handler
                checkbox.stateChanged.connect(self._on_input_changed)

    def _style_tabs(self) -> None:
        """Style the tab widget to make text bold and adjust height"""
        try:
            # Set style for the tab widget
            self.results_tabs.setStyleSheet(
                """
                QTabWidget::pane {
                    border: 1px solid #c0c0c0;
                }
                QTabBar::tab {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, (exact CSS coordinates)
                                               stop: 0 #e1e1e1, stop: 0.4 #dddddd,
                                               stop: 0.5 #d8d8d8, stop: 1.0 #d3d3d3);
                    border: 1px solid #c0c0c0;
                    border-bottom-color: #c2c7cb;
                    border-top-left-radius: 2px;
                    border-top-right-radius: 2px;
                    min-width: 8ex;
                    min-height: 18px;
                    max-height: 24px;
                    padding: 4px 10px;
                    font-weight: bold;
                    font-size: 10pt;
                }
                QTabBar::tab:selected, QTabBar::tab:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, (exact CSS coordinates)
                                               stop: 0 #fafafa, stop: 0.4 #f4f4f4,
                                               stop: 0.5 #e7e7e7, stop: 1.0 #fafafa);
                }
                QTabBar::tab:selected {
                    border-color: #9B9B9B;
                    border-bottom-color: #c2c7cb;
                }
                QTabBar::tab:!selected {
                    margin-top: 2px;
                }
            """
            )
        except (RuntimeError, AttributeError) as e:
            logger.exception("Error styling tabs: %s", e)

    def _create_visualization_tab(self) -> None:
        """Create visualization tab with charts and 3D plot only
        (controls moved to separate panel)"""
        self.viz_widget: QWidget | None = None  # Initialize with type hint
        self.loading_label: QLabel | None = None  # Initialize with type hint

        self.viz_widget = QWidget()
        viz_layout = QHBoxLayout(self.viz_widget)

        # Create loading indicator first
        self.loading_label = QLabel("Initializing 3D Visualization...")
        self.loading_label.setStyleSheet(
            """
            QLabel {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 40px;
                font-size: 14px;
                font-weight: bold;
                color: #495057;
            }
        """
        )
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viz_layout.addWidget(self.loading_label)

        # Initialize matplotlib widgets as None - will be created on first use
        self.current_fig: Figure | None = None
        self.current_canvas: FigureCanvas | None = None
        self.current_ax: Any | None = None
        self.power_fig: Figure | None = None
        self.power_canvas: FigureCanvas | None = None
        self.power_ax: Any | None = None
        self.electrode_fig: Figure | None = None
        self.electrode_canvas: FigureCanvas | None = None
        self.electrode_ax: Any | None = None
        self.matplotlib_initialized = False

        # Schedule matplotlib initialization after a short delay
        QTimer.singleShot(100, self._initialize_matplotlib_widgets)

    def _initialize_matplotlib_widgets(self) -> None:
        """Initialize matplotlib widgets with loading indicator"""
        try:
            if self.viz_widget is None:
                return
            # Get the layout before removing the loading label
            layout = self.viz_widget.layout()
            if layout is None:
                return
            viz_layout = cast(QHBoxLayout, layout)

            # Remove loading indicator
            if hasattr(self, "loading_label") and self.loading_label:
                self.loading_label.setParent(None)
                # self.loading_label = None # Don't set to None to avoid type error

            # Create visual builder
            builder = VisualizationBuilder(self)
            viz_content = builder.create_visualization_tab(
                scroll_callback=self._on_scroll,
                mouse_press_callback=self._on_mouse_press,
                mouse_release_callback=self._on_mouse_release,
                mouse_motion_callback=self._on_mouse_motion,
            )

            # Add viz content to existing layout
            # Set margins to 0 to avoid double padding
            viz_layout.setContentsMargins(0, 0, 0, 0)
            viz_layout.addWidget(viz_content)

            # Update references using dict unpacking or manual assignment
            widgets = builder.get_visualization_widgets()
            self.current_fig = widgets["current_fig"]
            self.current_canvas = widgets["current_canvas"]
            self.current_ax = widgets["current_ax"]
            self.power_fig = widgets["power_fig"]
            self.power_canvas = widgets["power_canvas"]
            self.power_ax = widgets["power_ax"]
            self.electrode_fig = widgets["electrode_fig"]
            self.electrode_canvas = widgets["electrode_canvas"]
            self.electrode_ax = widgets["electrode_ax"]

            self.matplotlib_initialized = True
            logger.info("Matplotlib widgets created successfully.")

            # Trigger initial calculation and visualization
            QTimer.singleShot(50, self._calculate_system)

        except (RuntimeError, AttributeError) as e:
            logger.exception("Error creating matplotlib widgets: %s", e)
            # Create fallback placeholder
            if viz_layout:
                placeholder = QLabel("3D Visualization (Matplotlib Error)")
                placeholder.setStyleSheet(
                    "background-color: #f0f0f0; border: 1px solid #ccc; padding: 20px;"
                )
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                viz_layout.addWidget(placeholder)

    def _create_results_tab(self) -> None:
        """Create combined results and analysis tab"""
        self.results_widget = QWidget()
        results_layout = QVBoxLayout(self.results_widget)

        # Create horizontal layout to use space better
        main_horizontal = QHBoxLayout()

        # Left column - Tables
        left_column = QVBoxLayout()

        # Resistance table
        res_group = QGroupBox("Phase Resistances and Current Paths")
        res_layout = QVBoxLayout(res_group)

        self.resistance_table = QTableWidget(3, 5)
        self.resistance_table.setHorizontalHeaderLabels(
            ["Phase", "Direct Glass (Ω)", "Via Metal (Ω)", "Total (Ω)", "Current Split"]
        )
        header = self.resistance_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        # Force smaller sizeHint to allow flexible resizing
        self.resistance_table.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        res_layout.addWidget(self.resistance_table)
        left_column.addWidget(res_group)

        # Power balance table
        power_group = QGroupBox("Power Distribution")
        power_layout = QVBoxLayout(power_group)

        self.power_table = QTableWidget(3, 4)
        self.power_table.setHorizontalHeaderLabels(
            ["Phase", "Power (kW)", "Balance (%)", "Temperature (°C)"]
        )
        header = self.power_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        # Force smaller sizeHint to allow flexible resizing
        self.power_table.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        power_layout.addWidget(self.power_table)
        left_column.addWidget(power_group)

        main_horizontal.addLayout(left_column)

        # Right column - Analysis
        right_column = QVBoxLayout()

        # Current path analysis
        path_group = QGroupBox("Current Path Analysis")
        path_layout = QFormLayout(path_group)

        self.path_labels: dict[str, QLineEdit] = {}
        metrics = [
            "Direct Glass Fraction",
            "Via Metal Fraction",
            "Path Resistance Ratio",
            "Thermal Efficiency",
        ]

        for metric in metrics:
            label = QLineEdit("N/A")
            label.setReadOnly(True)
            label.setStyleSheet("background-color: #f0f0f0;")
            path_layout.addRow(metric + ":", label)
            self.path_labels[metric] = label

        right_column.addWidget(path_group)

        # System metrics
        metrics_group = QGroupBox("System Metrics")
        metrics_layout = QFormLayout(metrics_group)

        self.metric_labels: dict[str, QLineEdit] = {}
        system_metrics = [
            "Total Power",
            "Power Balance",
            "Average Temperature",
            "Resistance Uniformity",
        ]

        for metric in system_metrics:
            label = QLineEdit("N/A")
            label.setReadOnly(True)
            label.setStyleSheet("background-color: #f0f0f0;")
            metrics_layout.addRow(metric + ":", label)
            self.metric_labels[metric] = label

        right_column.addWidget(metrics_group)
        right_column.addStretch()

        main_horizontal.addLayout(right_column)
        results_layout.addLayout(main_horizontal)

        # Recommendations (full width at bottom)
        rec_group = QGroupBox("Optimization Recommendations")
        rec_layout = QVBoxLayout(rec_group)

        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        # self.recommendations_text.setMaximumHeight(150)  # REMOVED: Allow flexible sizing
        rec_layout.addWidget(self.recommendations_text)
        results_layout.addWidget(rec_group)
