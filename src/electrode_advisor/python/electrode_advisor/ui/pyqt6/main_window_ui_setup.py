"""UI setup mixin for the ElectrodeAdvisorWidget.

Contains _init_ui, _create_status_panel, _create_visualization_tab,
_initialize_matplotlib_widgets, _create_results_tab, _style_tabs,
and _connect_checkbox_signals.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QTimer,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class UISetupMixin:
    """Mixin providing UI initialization for ElectrodeAdvisorWidget."""

    def _init_ui(self) -> None:
        """Initialize the user interface with scroll areas to prevent cutoff"""
        main_layout = QVBoxLayout(self)  # type: ignore[arg-type]

        # --- Toolbar for manual state management ---
        toolbar_layout = QHBoxLayout()
        save_btn = QPushButton("Save State")
        save_btn.setToolTip("Save the current Electrode Advisor state")
        save_btn.clicked.connect(self.save_state)  # type: ignore[attr-defined]
        load_btn = QPushButton("Load State")
        load_btn.setToolTip("Load the saved Electrode Advisor state")
        load_btn.clicked.connect(self.load_state)  # type: ignore[attr-defined]
        toolbar_layout.addWidget(save_btn)
        toolbar_layout.addWidget(load_btn)
        toolbar_layout.addStretch(1)
        main_layout.addLayout(toolbar_layout)

        # Create scroll area for the entire content
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create main content widget
        main_content = QWidget()
        content_layout = QVBoxLayout(main_content)

        # Create primary splitter for three panels
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(6)
        main_splitter.setChildrenCollapsible(False)

        # Left panel - inputs with its own scroll area
        self._create_input_panel()  # type: ignore[attr-defined]
        left_scroll = QScrollArea()
        left_scroll.setWidget(self.input_panel)  # type: ignore[attr-defined]
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        main_splitter.addWidget(left_scroll)

        # Center panel - results, visualization, and status
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        # Status panel
        self._create_status_panel()
        center_layout.addWidget(self.status_panel)  # type: ignore[attr-defined]

        # Tab widget for visualizations and results
        self.results_tabs = QTabWidget()

        # Visualization tab (charts + 3D plot only, no controls)
        self._create_visualization_tab()
        self.results_tabs.addTab(self.viz_widget, "AC Electrodes")  # type: ignore[attr-defined]

        # Results tab
        self._create_results_tab()
        self.results_tabs.addTab(self.results_widget, "Analysis")  # type: ignore[attr-defined]

        center_layout.addWidget(self.results_tabs)
        main_splitter.addWidget(center_widget)

        # Right panel - visual controls only
        self._create_visual_controls_panel()  # type: ignore[attr-defined]
        main_splitter.addWidget(self.visual_controls_panel)  # type: ignore[attr-defined]

        # Set stretch factors for three panels
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)

        # Style the splitter handle
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #d0d0d0;
                border: 1px solid #a0a0a0;
                margin: 1px;
            }
            QSplitter::handle:hover {
                background-color: #b0b0b0;
            }
            QSplitter::handle:pressed {
                background-color: #909090;
            }
            QSplitter::handle:horizontal {
                width: 6px;
            }
        """)

        content_layout.addWidget(main_splitter)
        main_scroll.setWidget(main_content)
        main_layout.addWidget(main_scroll)

        # Style the tabs
        self._style_tabs()

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
                try:
                    checkbox.stateChanged.disconnect()
                except (TypeError, RuntimeError) as disconnect_error:
                    logger.debug(
                        "Checkbox '%s' disconnect skipped: %s",
                        checkbox_name,
                        disconnect_error,
                    )
                checkbox.stateChanged.connect(self._on_input_changed)  # type: ignore[attr-defined]

    def _style_tabs(self) -> None:
        """Style the tab widget to make text bold and adjust height"""
        try:
            self.results_tabs.setStyleSheet("""
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
            """)
        except (RuntimeError, AttributeError) as e:
            logger.exception("Error styling tabs: %s", e)

    def _create_status_panel(self) -> None:
        """Create compact status panel"""
        self.status_panel = QGroupBox("Status")
        status_layout = QVBoxLayout(self.status_panel)
        status_layout.setContentsMargins(5, 5, 5, 5)

        self.status_label = QLabel("System Ready")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

    def _create_visualization_tab(self) -> None:
        """Create visualization tab with charts and 3D plot only"""
        self.viz_widget: QWidget | None = None
        self.loading_label: QLabel | None = None

        self.viz_widget = QWidget()
        viz_layout = QHBoxLayout(self.viz_widget)

        # Create loading indicator first
        self.loading_label = QLabel("Initializing 3D Visualization...")
        self.loading_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 40px;
                font-size: 14px;
                font-weight: bold;
                color: #495057;
            }
        """)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viz_layout.addWidget(self.loading_label)

        # Initialize matplotlib widgets as None
        self.current_fig: Any | None = None
        self.current_canvas: Any | None = None
        self.current_ax: Any | None = None
        self.power_fig: Any | None = None
        self.power_canvas: Any | None = None
        self.power_ax: Any | None = None
        self.electrode_fig: Any | None = None
        self.electrode_canvas: Any | None = None
        self.electrode_ax: Any | None = None
        self.matplotlib_initialized = False

        # Schedule matplotlib initialization after a short delay
        QTimer.singleShot(100, self._initialize_matplotlib_widgets)

    def _initialize_matplotlib_widgets(self) -> None:
        """Initialize matplotlib widgets with loading indicator"""
        try:
            if self.viz_widget is None:
                return
            layout = self.viz_widget.layout()
            if layout is None:
                return
            viz_layout = cast(QHBoxLayout, layout)

            # Remove loading indicator
            if hasattr(self, "loading_label") and self.loading_label:
                self.loading_label.setParent(None)  # type: ignore[call-overload]

            from .visualization_builder import VisualizationBuilder

            builder = VisualizationBuilder(self)  # type: ignore[arg-type]
            viz_content = builder.create_visualization_tab(
                scroll_callback=self._on_scroll,  # type: ignore[attr-defined]
                mouse_press_callback=self._on_mouse_press,  # type: ignore[attr-defined]
                mouse_release_callback=self._on_mouse_release,  # type: ignore[attr-defined]
                mouse_motion_callback=self._on_mouse_motion,  # type: ignore[attr-defined]
            )

            viz_layout.setContentsMargins(0, 0, 0, 0)
            viz_layout.addWidget(viz_content)

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

            QTimer.singleShot(50, self._calculate_system)  # type: ignore[attr-defined]

        except (RuntimeError, AttributeError) as e:
            logger.exception("Error creating matplotlib widgets: %s", e)
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

        # Recommendations
        rec_group = QGroupBox("Optimization Recommendations")
        rec_layout = QVBoxLayout(rec_group)

        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        rec_layout.addWidget(self.recommendations_text)
        results_layout.addWidget(rec_group)
