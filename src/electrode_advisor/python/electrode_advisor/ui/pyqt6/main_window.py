#!/usr/bin/env python3
"""
AC Electrode Advancement Module v5.1
Updated with corrected conductive path model:
- Electrodes conduct along their ENTIRE LENGTH within glass bath only
- Direct glass paths: Trapezoidal prisms formed by electrode lengths in glass
- Via-metal paths: Rectangular extrusions from electrode lengths in glass
- Accurate physics-based resistance calculations
- Fixed transparency controls
- Added metal vessel shell visualization
- Conductive paths properly constrained to glass bath
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

# Set environment variable before any Qt imports
os.environ["QT_API"] = "pyqt6"

# Set matplotlib backend to PyQt6 BEFORE any other imports
if os.environ.get("HEADLESS", "false").lower() == "true":
    try:
        mpl.use("Agg")
    except (ImportError, ValueError):
        pass  # Already using correct backend or unavailable
else:
    try:
        mpl.use("QtAgg")
    except (ImportError, ValueError):
        mpl.use("Agg")  # Fallback to non-interactive backend

# Prevent matplotlib from opening separate windows
mpl.rcParams["interactive"] = False  # Disable interactive mode

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
else:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_agg import (  # noqa: E402
            FigureCanvasAgg as FigureCanvas,
        )

from matplotlib.figure import Figure  # noqa: E402
from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal, pyqtSlot  # noqa: E402
from PyQt6.QtGui import QCloseEvent, QFont, QIcon  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from upstream_drift_tools.calculators.electrical import (  # noqa: E402
    ElectrodeConfig,
    GlassPropertiesInterface,
    ThreePhaseElectricalModelEnhanced,
)

from ...configs.color_schemes import (  # noqa: E402
    get_color_scheme,
)
from ...configs.ui_defaults import (  # noqa: E402
    PERIODIC_UPDATE_MS,
)
from ...configs.view_presets import (  # noqa: E402
    DEFAULT_Z_SCALE_FACTOR,
    DEFAULT_ZOOM_SCALE_FACTOR,
    get_view_preset,
)
from ...ui_builders.visualization_builder import VisualizationBuilder  # noqa: E402
from ...utils.visualization import ElectrodeVisualization  # noqa: E402

logger = logging.getLogger(__name__)

# BaseCalculatorWidget not available in standalone Tools context
BASE_CALCULATOR_AVAILABLE = False

# State management mixin not available in standalone Tools context
STATE_MIXIN_AVAILABLE = False

from .main_window_drawing import DrawingMixin  # noqa: E402
from .main_window_glass_integration import GlassIntegrationMixin  # noqa: E402
from .main_window_input_panel import InputPanelMixin  # noqa: E402
from .main_window_results_charts import ResultsAndChartsMixin  # noqa: E402
from .main_window_visual_controls import VisualControlsMixin  # noqa: E402
from .main_window_visualization_update import VisualizationUpdateMixin  # noqa: E402


# --- Main Widget for Tab Integration ---
class ElectrodeAdvisorWidget(
    GlassIntegrationMixin,
    ResultsAndChartsMixin,
    VisualizationUpdateMixin,
    VisualControlsMixin,
    InputPanelMixin,
    DrawingMixin,
    QWidget,
):
    """Main widget that can be embedded as a tab in another application"""

    # Signals for external communication
    data_updated = pyqtSignal(dict)
    optimization_complete = pyqtSignal(dict)
    glass_properties_requested = pyqtSignal()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Handle cleanup on close"""
        # Stop timers
        if hasattr(self, "calc_timer") and self.calc_timer.isActive():
            self.calc_timer.stop()

        self.save_state()
        if event:
            event.accept()
            super().closeEvent(event)

    def __init__(
        self,
        config: ElectrodeConfig | None = None,
        glass_interface: GlassPropertiesInterface | None = None,
        parent: QWidget | None = None,
        calculator_name: str = "ElectrodeAdvisor",
        # Default configuration
        vertical_spreading: float = 1.0,
        horizontal_spreading: float = 1.0,
    ) -> None:
        """Initialize the ElectrodeAdvisor widget.

        Args:
            config: Electrode configuration object
            glass_interface: Glass properties interface
            parent: Parent widget
            calculator_name: Name for state management
            vertical_spreading: Vertical spreading factor for conductive paths.
            horizontal_spreading: Horizontal spreading factor for conductive paths.
        """
        super().__init__(parent)

        # Store calculator name for state management
        self.calculator_name = calculator_name

        # Set window icon
        try:
            # Path relative to this file: ../icons/ElectrodeAdvisor_Icon.jpg
            # Icon directory path relative to this file: ../icons/
            icon_path = (
                Path(os.path.dirname(os.path.abspath(__file__))).parent
                / "icons"
                / "ElectrodeAdvisor_Icon.jpg"
            )
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to set window icon: {e}")

        # Initialize electrode advisor widget
        logger.info(
            "ElectrodeAdvisorWidget v5.1 loaded from electrode/electrode_advisor.py",
        )

        # Initialization flag to prevent calculations during setup
        self._initialization_complete = False

        # Configuration
        self.config = config or ElectrodeConfig()
        self.glass_interface = glass_interface or GlassPropertiesInterface()

        # Visual settings
        self.current_color_scheme = "Default"

        # Models
        self.electrical_model = ThreePhaseElectricalModelEnhanced(
            self.config, self.glass_interface
        )

        # Visualization utility
        self.visualizer = ElectrodeVisualization()

        # Data storage - Add type hints
        self.current_measurements: dict[str, Any] = {}
        self.calculation_results: dict[str, Any] = {}
        self.optimization_results: dict[str, Any] = {}

        # Initialize UI components to None or dummy to satisfy Mypy
        self.depth_inputs: dict[int, QDoubleSpinBox] = {}
        self.min_scale_input: QDoubleSpinBox | None = None
        self.max_scale_input: QDoubleSpinBox | None = None

        # Mouse interaction states
        self._mouse_pressed: bool = False
        self._last_mouse_pos: tuple[float, float] | None = None

        # Setup UI
        self._init_ui()
        self._setup_timers()

        # Mark initialization as complete
        self._initialization_complete = True

        # Connect all checkbox signals for proper updates
        self._connect_checkbox_signals()

        # Defer initial calculation until after the event loop starts
        QTimer.singleShot(100, self._calculate_system)

        # --- State management integration ---
        QTimer.singleShot(0, self.setup_state_management)
        QTimer.singleShot(0, self.load_state)

        # --- Setup context menu for manual state management ---
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def setup_state_management(self) -> None:
        """Register widgets for state management (stub for future extensibility)"""

    def save_state(self) -> None:
        """Save current state to persistent storage"""
        try:
            state_data = self.get_current_state()
            # Save to file using the calculator name as identifier

            from integrated_process_simulator.utilities.state_manager import (
                StateManager,
            )

            state_manager = StateManager()
            filename = f"{self.calculator_name}_state.json"
            state_manager.save_state(filename, state_data)
            logger.info("Electrode Advisor state saved to %s", filename)
        except ImportError as e:
            logger.warning("Warning: Could not save state: %s", e)

    def load_state(self) -> None:
        """Load state from persistent storage"""
        try:
            from integrated_process_simulator.utilities.state_manager import (
                StateManager,
            )

            state_manager = StateManager()
            filename = f"{self.calculator_name}_state.json"

            # Check if state file exists
            state_data = state_manager.load_state(filename)
            if state_data is not None:
                success = self.restore_state(state_data)
                if success:
                    logger.info("Electrode Advisor state loaded from %s", filename)
                else:
                    logger.error("Failed to restore state from %s", filename)
            else:
                logger.info("No saved state found for %s", filename)
        except ImportError as e:
            logger.warning("Warning: Could not load state: %s", e)

    def show_context_menu(self, position: QPoint) -> None:
        """Show context menu for manual state management"""
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self)

        # State management actions
        save_action = menu.addAction("Save State")
        if save_action:
            save_action.triggered.connect(lambda checked: self.save_state())

        load_action = menu.addAction("Load State")
        if load_action:
            load_action.triggered.connect(lambda checked: self.load_state())

        menu.addSeparator()

        # Copy actions
        copy_action = menu.addAction("Copy Results")
        if copy_action:
            copy_action.triggered.connect(lambda checked: self.copy_results())

        menu.exec(self.mapToGlobal(position))

    def copy_results(self) -> None:
        """Copy current results to clipboard"""
        try:
            from PyQt6.QtWidgets import QApplication

            # Get current calculation results
            if hasattr(self, "calculation_results") and self.calculation_results:
                results_text = "Electrode Advisor Results\n"
                results_text += (
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                # Add resistance data
                if "resistances" in self.calculation_results:
                    results_text += "Resistances:\n"
                    for path, resistance in self.calculation_results[
                        "resistances"
                    ].items():
                        results_text += f"  {path}: {resistance:.3f} Ω\n"

                # Add current data
                if "actual_currents" in self.calculation_results:
                    results_text += "\nCurrents:\n"
                    for path, current in self.calculation_results[
                        "actual_currents"
                    ].items():
                        results_text += f"  {path}: {current:.3f} A\n"

                # Copy to clipboard
                clipboard = QApplication.clipboard()
                if clipboard:
                    clipboard.setText(results_text)
                logger.info("Results copied to clipboard")
            else:
                logger.info("No results available to copy.")
        except ImportError as e:
            logger.warning("Warning: Could not copy results: %s", e)

    def _init_ui(self) -> None:
        """Initialize the user interface with scroll areas to prevent cutoff"""
        main_layout = QVBoxLayout(self)

        # --- Toolbar for manual state management ---
        toolbar_layout = QHBoxLayout()
        save_btn = QPushButton("Save State")
        save_btn.setToolTip("Save the current Electrode Advisor state")
        save_btn.clicked.connect(self.save_state)
        load_btn = QPushButton("Load State")
        load_btn.setToolTip("Load the saved Electrode Advisor state")
        load_btn.clicked.connect(self.load_state)
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

        # Create primary splitter for three panels: Left (inputs) | Center (viz) | Right (controls)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(
            6
        )  # Make splitter handle more visible and easier to grab
        main_splitter.setChildrenCollapsible(
            False
        )  # Prevent panels from collapsing completely

        # Left panel - inputs with its own scroll area
        self._create_input_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(self.input_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # left_scroll.setMaximumWidth(450)  # REMOVED: Allow flexible sizing
        main_splitter.addWidget(left_scroll)

        # Center panel - results, visualization, and status
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        # Status panel
        self._create_status_panel()
        center_layout.addWidget(self.status_panel)

        # Tab widget for visualizations and results
        self.results_tabs = QTabWidget()

        # Visualization tab (charts + 3D plot only, no controls)
        self._create_visualization_tab()
        self.results_tabs.addTab(self.viz_widget, "AC Electrodes")

        # Results tab
        self._create_results_tab()
        self.results_tabs.addTab(self.results_widget, "Analysis")

        center_layout.addWidget(self.results_tabs)
        main_splitter.addWidget(center_widget)

        # Right panel - visual controls only
        self._create_visual_controls_panel()
        main_splitter.addWidget(self.visual_controls_panel)

        # Set initial splitter sizes and stretch factors for three panels
        # main_splitter.setSizes([350, 900, 350])  # REMOVED: Allow flexible sizing
        main_splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        main_splitter.setStretchFactor(1, 1)  # Center panel gets most extra space
        main_splitter.setStretchFactor(2, 0)  # Right panel has fixed preferred size

        # Style the splitter handle to make it more visible and easier to drag
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

    # Input panel methods moved to InputPanelMixin (main_window_input_panel.py)

    def _create_status_panel(self) -> None:
        """Create compact status panel"""
        self.status_panel = QGroupBox("Status")
        status_layout = QVBoxLayout(self.status_panel)
        # self.status_panel.setMaximumHeight(60)  # REMOVED: Allow flexible sizing
        status_layout.setContentsMargins(5, 5, 5, 5)  # Tighter margins

        self.status_label = QLabel("System Ready")
        self.status_label.setFont(QFont("Arial", 10))  # Smaller font
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)

        # Progress bar for calculations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        # self.progress_bar.setMaximumHeight(15)  # REMOVED: Allow flexible sizing
        status_layout.addWidget(self.progress_bar)

    def _create_visualization_tab(self) -> None:
        """Create visualization tab with charts and 3D plot only
        (controls moved to separate panel)"""
        self.viz_widget: QWidget | None = None  # Initialize with type hint
        self.loading_label: QLabel | None = None  # Initialize with type hint

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

    # Visual controls methods moved to VisualControlsMixin (main_window_visual_controls.py)

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

    def _setup_timers(self) -> None:
        """Setup update timers"""
        self.calc_timer = QTimer()
        self.calc_timer.timeout.connect(self._periodic_update)
        self.calc_timer.start(PERIODIC_UPDATE_MS)

    @pyqtSlot()
    def _on_input_changed(self) -> None:
        """Handle input parameter changes"""
        if getattr(self, "_initialization_complete", False):
            # Only call _calculate_system() which internally calls _update_3d_visualization()
            # Remove the duplicate _draw_3d_real_geometry() call that was causing conflicts
            self._calculate_system()

    @pyqtSlot()
    def _on_zoom_slider_changed(self) -> None:
        """Handle zoom slider changes"""
        zoom_value = self.zoom_slider.value()
        self.zoom_label.setText(f"Zoom: {zoom_value}%")

        # Apply zoom to 3D plot
        if self.electrode_ax is not None:
            # Get current view limits
            xlim = self.electrode_ax.get_xlim()
            ylim = self.electrode_ax.get_ylim()

            # Calculate center
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2

            # Calculate base range (at 100% zoom)
            base_range = (
                self.bath_diameter_input.value() / 2
                + self.electrode_extension_slider.value()
            )

            # Apply zoom factor
            zoom_factor = zoom_value / 100.0
            new_range = base_range / zoom_factor * 1.1  # 1.1 for margin

            # Set new limits
            self.electrode_ax.set_xlim(
                x_center - new_range / 2, x_center + new_range / 2
            )
            self.electrode_ax.set_ylim(
                y_center - new_range / 2, y_center + new_range / 2
            )
            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()
            if hasattr(self.electrode_ax, "set_zlim"):
                z_range = (
                    (
                        self.glass_layer_height_input.value()
                        + self.metal_layer_height_input.value()
                    )
                    / zoom_factor
                    * 1.2
                )
                self.electrode_ax.set_zlim(0, z_range)

            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()

    @pyqtSlot()
    def _validate_glass_height(self) -> None:
        """Validate that glass height is above electrode tips"""
        try:
            glass_height = self.glass_layer_height_input.value()
            metal_height = self.metal_layer_height_input.value()

            # Get maximum electrode depth
            max_electrode_depth: float = 0.0
            for i in range(3):
                if i in self.depth_inputs:
                    depth = self.depth_inputs[i].value()
                    max_electrode_depth = max(max_electrode_depth, depth)

            # Calculate minimum glass height needed (should be above electrode tips)
            min_glass_height = (
                max_electrode_depth + metal_height + 1.0
            )  # 1 inch safety margin

            if glass_height < min_glass_height:
                # Show warning and suggest minimum height
                QMessageBox.warning(
                    self,
                    "Glass Height Warning",
                    f"Glass height ({glass_height:.1f} in) should be above electrode tips.\n"
                    f"Minimum recommended height: {min_glass_height:.1f} in\n"
                    f"(Electrode depth: {max_electrode_depth:.1f} in + "
                    f"Metal height: {metal_height:.1f} in + Safety margin: 1.0 in)",
                )

                # Automatically adjust to minimum safe height
                self.glass_layer_height_input.setValue(min_glass_height)

            # Continue with normal calculation
            self._calculate_system()

        except (RuntimeError, AttributeError) as e:
            logger.exception("Error in glass height validation: %s", e)
            # Fall back to normal calculation
            self._calculate_system()

    @pyqtSlot()
    def _on_metal_conductivity_changed(self) -> None:
        """Handle metal layer conductivity toggle"""
        try:
            is_enabled = self.metal_conductive_checkbox.isChecked()
            logger.debug(
                "[DEBUG] Metal layer conductivity: %s",
                "Enabled" if is_enabled else "Disabled",
            )

            # If metal conductivity is disabled, hide metal layer in visualization
            if hasattr(self, "show_metal_checkbox"):
                if not is_enabled:
                    # Disable metal layer visualization when conductivity is off
                    self.show_metal_checkbox.setChecked(False)
                    # Optionally disable the checkbox to prevent manual enabling
                    self.show_metal_checkbox.setEnabled(False)
                    # Add visual indication that it's disabled
                    self.show_metal_checkbox.setStyleSheet(
                        "QCheckBox { color: #888888; }"
                    )
                    self.show_metal_checkbox.setToolTip(
                        "Metal layer visualization disabled when conduction is off"
                    )
                else:
                    # Re-enable metal layer visualization controls
                    self.show_metal_checkbox.setEnabled(True)
                    self.show_metal_checkbox.setChecked(True)
                    self.show_metal_checkbox.setStyleSheet("")
                    self.show_metal_checkbox.setToolTip("")

            # Recalculate system with new conductivity model
            self._calculate_system()

            # Update results tables to reflect new percentages
            self._update_results_tables()
            self._update_analysis_display()

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error handling metal conductivity change: %s", e)
            import traceback

            traceback.print_exc()

    def _calculate_system(self) -> None:
        """Calculate System method.

        Returns:
            None
        """
        try:
            logger.debug("[DEBUG] _calculate_system called")
            # Update configuration with current spreading factors
            self.config.vertical_spreading_factor = (
                self.vertical_spreading_input.value()
            )
            self.config.horizontal_spreading_factor = (
                self.horizontal_spreading_input.value()
            )

            # Read all input values
            depths = np.array(
                [
                    self.depth_inputs[0].value(),
                    self.depth_inputs[1].value(),
                    self.depth_inputs[2].value(),
                ]
            )

            bath_diameter = self.bath_diameter_input.value()
            electrode_diameter = float(self.electrode_diameter_combo.currentText())
            metal_layer_height = self.metal_layer_height_input.value()
            bath_temperature = self.bath_temp_input.value()

            # Voltages from phase inputs
            voltages = np.array(
                [
                    cast(QDoubleSpinBox, self.phase_inputs["1-2"]["voltage"]).value(),
                    cast(QDoubleSpinBox, self.phase_inputs["2-3"]["voltage"]).value(),
                    cast(QDoubleSpinBox, self.phase_inputs["3-1"]["voltage"]).value(),
                ]
            )

            # K factors
            k_factors = {
                "K_tt": self.k_tt_input.value() * self.config.k_scaling_factor,
                "K_vert": self.k_vert_input.value() * self.config.k_scaling_factor,
            }

            # Calculate system state using the electrical model
            conductive_height = self.conductive_layer_height_input.value()
            self.calculation_results = self.electrical_model.calculate_system_state(
                depths=depths,
                bath_diameter=bath_diameter,
                tip_diameter=electrode_diameter,
                metal_depth=metal_layer_height,
                k_factors=k_factors,
                bath_temperature=bath_temperature,
                voltages=voltages,
                conductive_height=conductive_height,
            )
            logger.debug("[DEBUG] calculation_results: %s", self.calculation_results)

            # Update displays
            self._update_3d_visualization()
            self._update_temperature_profile()
            self._update_current_distribution()
            self._update_power_distribution()
            self._update_results_tables()
            self._update_analysis_display()
            self._update_status("Calculation completed successfully", "ok")

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            # Handle calculation errors gracefully
            error_msg = f"Calculation error: {e!s}"
            logger.exception(error_msg)
            self._update_status(error_msg, "error")

    def _update_status(self, message: str, status_type: str = "ok") -> None:
        """Update status display"""
        self.status_label.setText(message)

        if self.config.colors is None:
            return

        color_map = {
            "ok": self.config.colors["status_ok"],
            "warn": self.config.colors["status_warn"],
            "error": self.config.colors["status_err"],
        }

        color = color_map.get(status_type, self.config.colors["status_ok"])
        # Colors may be hex strings (shared engine) or QColor objects
        color_str = color.name() if hasattr(color, "name") else str(color)
        self.status_panel.setStyleSheet(f"background-color: {color_str}")

    # Results/charts update methods moved to ResultsAndChartsMixin
    # (main_window_results_charts.py)

    @pyqtSlot()
    def _run_optimization(self) -> None:
        """Run electrode position optimization"""
        # This would implement the optimization algorithm
        # For now, just show a message
        QMessageBox.information(
            self,
            "Optimization",
            "Optimization feature will be implemented with full algorithm integration",
        )

        # Emit optimization complete signal
        self.optimization_complete.emit({"status": "pending"})

    def _periodic_update(self) -> None:
        """Periodic update for real-time monitoring"""
        # This could refresh data from external sources

    # Mouse interaction handlers
    def _on_scroll(self, event: Any) -> None:
        """Handle mouse scroll for zoom"""
        try:
            if event.inaxes == self.electrode_ax:
                # Zoom factor
                zoom_factor = 1.1 if event.button == "down" else 0.9

                # Update zoom slider
                current_zoom = self.zoom_slider.value()
                new_zoom = int(current_zoom * zoom_factor)
                new_zoom = max(50, min(200, new_zoom))  # Clamp to range
                self.zoom_slider.setValue(new_zoom)

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error in scroll handler: %s", e)

    def _on_mouse_press(self, event: Any) -> None:
        """Handle mouse press for rotation and pan"""
        try:
            if event.inaxes == self.electrode_ax:
                self._mouse_pressed = True
                # Use pixel coordinates for consistent behavior
                self._last_mouse_pos = (
                    (event.x, event.y)
                    if event.x is not None and event.y is not None
                    else None
                )
                # Get 3D view angles for rotation mode
                if hasattr(self.electrode_ax, "elev") and hasattr(
                    self.electrode_ax, "azim"
                ):
                    self._last_elev = self.electrode_ax.elev
                    self._last_azim = self.electrode_ax.azim
                else:
                    # Fallback: use default angles if not available
                    self._last_elev = 20
                    self._last_azim = 45
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error in mouse press handler: %s", e)

    def _on_mouse_release(self, event: Any) -> None:
        """Handle mouse release"""
        try:
            self._mouse_pressed = False
            logger.info("Mouse released")
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error in mouse release handler: %s", e)

    def _on_mouse_motion(self, event: Any) -> None:
        """Handle mouse motion for rotation or panning based on interaction mode"""
        try:
            if (
                self._mouse_pressed
                and event.inaxes == self.electrode_ax
                and self._last_mouse_pos
                and event.x is not None
                and event.y is not None
            ):
                dx = event.x - self._last_mouse_pos[0]
                dy = event.y - self._last_mouse_pos[1]
                logger.info(
                    "Mouse motion: dx=%s, dy=%s, mode=%s", dx, dy, self.interaction_mode
                )
                if self.interaction_mode == "rotation":
                    # Only rotate if in rotation mode
                    if hasattr(self.electrode_ax, "view_init") and hasattr(
                        self, "_last_elev"
                    ):
                        rotation_sensitivity = 0.5
                        new_elev = self._last_elev + dy * rotation_sensitivity
                        new_azim = self._last_azim + dx * rotation_sensitivity
                        new_elev = max(-90, min(90, new_elev))
                        self.electrode_ax.view_init(elev=new_elev, azim=new_azim)
                        if self.electrode_canvas:
                            self.electrode_canvas.draw_idle()
                        self._last_elev = new_elev
                        self._last_azim = new_azim
                elif self.interaction_mode == "pan":
                    # Only pan if in pan mode
                    if hasattr(self.electrode_ax, "get_xlim"):
                        x_range = self.electrode_ax.get_xlim()
                        y_range = self.electrode_ax.get_ylim()
                        pan_factor = 0.01
                        x_shift = dx * pan_factor * (x_range[1] - x_range[0])
                        y_shift = dy * pan_factor * (y_range[1] - y_range[0])
                        self.electrode_ax.set_xlim(
                            x_range[0] - x_shift, x_range[1] - x_shift
                        )
                        self.electrode_ax.set_ylim(
                            y_range[0] + y_shift, y_range[1] + y_shift
                        )
                    if self.electrode_canvas is not None:
                        self.electrode_canvas.draw_idle()
                self._last_mouse_pos = (event.x, event.y)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error in mouse motion handler: %s", e)

    def _on_interaction_mode_changed(self) -> None:
        """Handle interaction mode change"""
        try:
            if self.rotation_mode_radio.isChecked():
                self.interaction_mode = "rotation"
            else:
                self.interaction_mode = "pan"
            logger.info("3D interaction mode changed to: %s", self.interaction_mode)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error changing interaction mode: %s", e)

    def _reset_3d_view(self) -> None:
        """Reset 3D view to default angle and zoom"""
        try:
            if self.electrode_canvas is None or self.electrode_ax is None:
                return

            if hasattr(self.electrode_ax, "view_init"):
                self.electrode_ax.view_init(elev=20, azim=45)

            # Reset zoom slider to 100%
            if hasattr(self, "zoom_slider"):
                self.zoom_slider.setValue(100)

            self.electrode_canvas.draw_idle()
            logger.info("3D view reset to default")
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error resetting 3D view: %s", e)

    # Public methods for external integration
    def _set_view_preset(self, preset: str) -> None:
        """Set predefined view angles using view presets from config."""
        if self.electrode_ax is None:
            return

        try:
            view_angle = get_view_preset(preset)
            self.electrode_ax.view_init(elev=view_angle.elev, azim=view_angle.azim)

            # For default view, also reset pan/zoom
            if preset == "default":
                bath_diameter = self.bath_diameter_input.value()
                extension_length = float(self.electrode_extension_slider.value())
                glass_height = self.glass_layer_height_input.value()
                metal_height = self.metal_layer_height_input.value()
                max_range = max(
                    bath_diameter / 2 + extension_length, glass_height + metal_height
                )
                zoom_factor = self.zoom_slider.value() / 100.0
                scaled_range = max_range / zoom_factor * DEFAULT_ZOOM_SCALE_FACTOR
                self.electrode_ax.set_xlim(-scaled_range, scaled_range)
                self.electrode_ax.set_ylim(-scaled_range, scaled_range)
                if hasattr(self.electrode_ax, "set_zlim"):
                    self.electrode_ax.set_zlim(
                        0,
                        (glass_height + metal_height)
                        / zoom_factor
                        * DEFAULT_Z_SCALE_FACTOR,
                    )

            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error setting view preset: %s", e)

    def _on_color_scheme_changed(self, scheme: str) -> None:
        """Handle color scheme changes"""
        try:
            self.current_color_scheme = scheme
            # Trigger visualization update with new color scheme
            self._on_input_changed()
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error changing color scheme: %s", e)

    def _get_color_scheme_colors(self) -> list[str]:
        """Get colors based on current color scheme."""
        return get_color_scheme(self.current_color_scheme)

    def _get_transparency_values(self) -> dict[str, float]:
        """Get current transparency values from sliders"""
        return {
            "electrodes": self.electrode_alpha_slider.value() / 100.0,
            "glass": self.glass_alpha_slider.value() / 100.0,
            "metal": self.metal_alpha_slider.value() / 100.0,
            "paths": self.path_alpha_slider.value() / 100.0,
            "refractory": self.refractory_alpha_slider.value() / 100.0,
            "metal_shell": self.metal_shell_alpha_slider.value() / 100.0,
        }

    def get_current_state(self) -> dict[str, Any]:
        """Get current state for state management"""
        try:
            return {
                "calculator_name": "ElectrodeAdvisor",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "bath_radius": self.bath_diameter_input.value() / 2.0,
                    "electrode_radius": float(
                        self.electrode_diameter_combo.currentText()
                    )
                    / 2.0,
                    "metal_height": self.config.metal_layer_height,
                    "glass_height": self.config.glass_depth,
                    "bath_temperature": self.config.bath_temperature_base,
                    "vertical_spreading": self.config.vertical_spreading_factor,
                    "horizontal_spreading": self.config.horizontal_spreading_factor,
                },
                "electrode_depths": {
                    "E1": (
                        getattr(self, "depth_inputs", [None, None, None])[0].value()
                        if hasattr(self, "depth_inputs") and len(self.depth_inputs) > 0
                        else 0
                    ),
                    "E2": (
                        getattr(self, "depth_inputs", [None, None, None])[1].value()
                        if hasattr(self, "depth_inputs") and len(self.depth_inputs) > 1
                        else 0
                    ),
                    "E3": (
                        getattr(self, "depth_inputs", [None, None, None])[2].value()
                        if hasattr(self, "depth_inputs") and len(self.depth_inputs) > 2
                        else 0
                    ),
                },
                "electrical_measurements": {
                    "1-2": {
                        "current": (
                            getattr(self, "phase_inputs", {})
                            .get("1-2", {})
                            .get("current", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "1-2" in self.phase_inputs
                            else 0
                        ),
                        "voltage": (
                            getattr(self, "phase_inputs", {})
                            .get("1-2", {})
                            .get("voltage", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "1-2" in self.phase_inputs
                            else 0
                        ),
                    },
                    "2-3": {
                        "current": (
                            getattr(self, "phase_inputs", {})
                            .get("2-3", {})
                            .get("current", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "2-3" in self.phase_inputs
                            else 0
                        ),
                        "voltage": (
                            getattr(self, "phase_inputs", {})
                            .get("2-3", {})
                            .get("voltage", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "2-3" in self.phase_inputs
                            else 0
                        ),
                    },
                    "3-1": {
                        "current": (
                            getattr(self, "phase_inputs", {})
                            .get("3-1", {})
                            .get("current", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "3-1" in self.phase_inputs
                            else 0
                        ),
                        "voltage": (
                            getattr(self, "phase_inputs", {})
                            .get("3-1", {})
                            .get("voltage", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "3-1" in self.phase_inputs
                            else 0
                        ),
                    },
                },
                "results": self.calculation_results,
            }
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error getting current state: %s", e)
            return {"calculator_name": "ElectrodeAdvisor", "error": str(e)}

    def restore_state(self, state_data: dict[str, Any]) -> bool:
        """Restore state from saved data"""
        try:
            if state_data.get("calculator_name") != "ElectrodeAdvisor":
                logger.info("State data mismatch for ElectrodeAdvisor")
                return False

            # Restore configuration
            if "config" in state_data:
                config_data = state_data["config"]
                if "bath_radius" in config_data:
                    self.bath_diameter_input.setValue(config_data["bath_radius"] * 2.0)
                if "electrode_radius" in config_data:
                    self.electrode_diameter_combo.setCurrentText(
                        str(config_data["electrode_radius"] * 2.0)
                    )
                if "metal_height" in config_data:
                    self.config.metal_layer_height = config_data["metal_height"]
                if "glass_height" in config_data:
                    self.config.glass_depth = config_data["glass_height"]
                if "bath_temperature" in config_data:
                    self.config.bath_temperature_base = config_data["bath_temperature"]
                if "vertical_spreading" in config_data:
                    self.config.vertical_spreading_factor = config_data[
                        "vertical_spreading"
                    ]
                if "horizontal_spreading" in config_data:
                    self.config.horizontal_spreading_factor = config_data[
                        "horizontal_spreading"
                    ]

            # Restore electrode depths
            if "electrode_depths" in state_data and hasattr(self, "depth_inputs"):
                depths_data = state_data["electrode_depths"]
                if "E1" in depths_data and len(self.depth_inputs) > 0:
                    self.depth_inputs[0].setValue(depths_data["E1"])
                if "E2" in depths_data and len(self.depth_inputs) > 1:
                    self.depth_inputs[1].setValue(depths_data["E2"])
                if "E3" in depths_data and len(self.depth_inputs) > 2:
                    self.depth_inputs[2].setValue(depths_data["E3"])

            # Restore electrical measurements
            if "electrical_measurements" in state_data and hasattr(
                self, "phase_inputs"
            ):
                measurements_data = state_data["electrical_measurements"]
                for phase in ["1-2", "2-3", "3-1"]:
                    if phase in measurements_data and phase in self.phase_inputs:
                        if "current" in measurements_data[phase]:
                            cast(
                                QDoubleSpinBox, self.phase_inputs[phase]["current"]
                            ).setValue(measurements_data[phase]["current"])
                        if "voltage" in measurements_data[phase]:
                            cast(
                                QDoubleSpinBox, self.phase_inputs[phase]["voltage"]
                            ).setValue(measurements_data[phase]["voltage"])

            # Trigger recalculation
            QTimer.singleShot(100, self._calculate_system)

            logger.info("State restored for ElectrodeAdvisor")
            return True

        except (RuntimeError, AttributeError) as e:
            logger.exception("Error restoring state for ElectrodeAdvisor: %s", e)
            return False

    def set_electrode_depths(self, depths: list[float]) -> None:
        """Set electrode depths programmatically"""
        for i, depth in enumerate(depths[:3]):
            self.depth_inputs[i].setValue(depth)

    def set_electrical_measurements(
        self, currents: list[float], voltages: list[float]
    ) -> None:
        """Set electrical measurements programmatically"""
        phases = ["1-2", "2-3", "3-1"]
        for i, phase in enumerate(phases[:3]):
            if phase in self.phase_inputs:
                cast(QDoubleSpinBox, self.phase_inputs[phase]["current"]).setValue(
                    currents[i]
                )
                cast(QDoubleSpinBox, self.phase_inputs[phase]["voltage"]).setValue(
                    voltages[i]
                )

    def export_results(self, filename: str) -> bool:
        """Export calculation results to file"""
        try:
            import json

            with open(filename, "w") as f:
                # Prepare exportable data
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "configuration": {
                        "bath_diameter": self.bath_diameter_input.value(),
                        "electrode_diameter": float(
                            self.electrode_diameter_combo.currentText()
                        ),
                        "metal_height": self.metal_layer_height_input.value(),
                        "glass_height": self.glass_layer_height_input.value(),
                        "bath_temperature": self.bath_temp_input.value(),
                        "vertical_spreading": self.config.vertical_spreading_factor,
                        "horizontal_spreading": self.config.horizontal_spreading_factor,
                    },
                    "electrode_depths": {
                        "E1": self.depth_inputs[0].value(),
                        "E2": self.depth_inputs[1].value(),
                        "E3": self.depth_inputs[2].value(),
                    },
                    "results": self.calculation_results,
                }
                json.dump(export_data, f, indent=2)
                logger.info("Results exported to %s", filename)
                return True
        except (PermissionError, OSError) as e:
            logger.exception("Error exporting results: %s", e)
            return False

    def _export_3d_plot(self) -> None:
        """Export the 3D plot to an image file"""
        try:
            from PyQt6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export 3D Plot",
                "electrode_3d_plot.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;JPG files (*.jpg)",
            )

            if filename and self.electrode_fig is not None:
                # Save the main 3D plot
                self.electrode_fig.savefig(filename, dpi=300, bbox_inches="tight")
                self._update_status(f"3D plot exported to {filename}", "ok")

        except ImportError as e:
            self._update_status(f"Error exporting 3D plot: {e!s}", "error")
            logger.exception("Export error: %s", e)

    def _export_charts(self) -> None:
        """Export the current and power charts to image files"""
        try:
            from PyQt6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Charts",
                "electrode_charts.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;JPG files (*.jpg)",
            )

            if filename:
                # Create a combined figure with both charts
                # Performance: Use Figure() directly instead of plt.figure() to avoid
                # pyplot state machine memory leaks in long-running sessions
                combined_fig = Figure(
                    figsize=(6, 3)
                )  # REDUCED: Smaller figure size to allow flexible sizing

                # Copy current chart
                ax1 = combined_fig.add_subplot(1, 2, 1)

                # Recreate current distribution chart
                phase_currents = [
                    self.phase_inputs["1-2"]["current"].value(),
                    self.phase_inputs["2-3"]["current"].value(),
                    self.phase_inputs["3-1"]["current"].value(),
                ]
                phases = ["1-2", "2-3", "3-1"]
                colors = ["#FF4444", "#44FF44", "#4444FF"]
                x = np.arange(len(phases))
                width = 0.35

                ax1.bar(
                    x - width / 2,
                    phase_currents,
                    width,
                    label="Phase Current",
                    color=colors,
                    alpha=0.8,
                )
                line_currents = [current * 0.8 for current in phase_currents]
                ax1.bar(
                    x + width / 2,
                    line_currents,
                    width,
                    label="Line Current",
                    color=colors,
                    alpha=0.5,
                )

                ax1.set_xlabel("Phase")
                ax1.set_ylabel("Current (A)")
                ax1.set_title("Current Distribution")
                ax1.set_xticks(x)
                ax1.set_xticklabels(phases)
                ax1.legend(loc="upper right")
                ax1.grid(True, alpha=0.3)

                # Copy power chart
                ax2 = combined_fig.add_subplot(1, 2, 2)

                # Recreate power distribution chart
                powers = []
                colors2 = ["#FF8C00", "#32CD32", "#1E90FF"]
                for phase in phases:
                    current = self.phase_inputs[phase]["current"].value()
                    voltage = self.phase_inputs[phase]["voltage"].value()
                    power = current * voltage / 1000
                    powers.append(power)

                ax2.bar(phases, powers, color=colors2, alpha=0.7)
                ax2.set_xlabel("Phase")
                ax2.set_ylabel("Power (kW)")
                ax2.set_title("Power Distribution")
                ax2.grid(True, alpha=0.3)

                combined_fig.tight_layout()
                combined_fig.savefig(filename, dpi=300, bbox_inches="tight")
                plt.close(combined_fig)

                self._update_status(f"Charts exported to {filename}", "ok")

        except ImportError as e:
            self._update_status(f"Error exporting charts: {e!s}", "error")
            logger.exception("Export error: %s", e)


# For backward compatibility with the wrapper
if __name__ == "__main__":
    # Test as standalone application
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = ElectrodeAdvisorWidget()
    widget.show()
    sys.exit(app.exec())
