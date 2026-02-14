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

from ...configs.ui_defaults import (  # noqa: E402
    PERIODIC_UPDATE_MS,
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
from .main_window_interaction import InteractionMixin  # noqa: E402
from .main_window_results_charts import ResultsAndChartsMixin  # noqa: E402
from .main_window_state_export import StateAndExportMixin  # noqa: E402
from .main_window_view_presets import ViewPresetsMixin  # noqa: E402
from .main_window_visual_controls import VisualControlsMixin  # noqa: E402
from .main_window_visualization_update import VisualizationUpdateMixin  # noqa: E402


# --- Main Widget for Tab Integration ---
class ElectrodeAdvisorWidget(
    StateAndExportMixin,
    ViewPresetsMixin,
    InteractionMixin,
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

    # Mouse/interaction methods -> InteractionMixin (main_window_interaction.py)
    # View presets/color scheme -> ViewPresetsMixin (main_window_view_presets.py)
    # State/export methods -> StateAndExportMixin (main_window_state_export.py)


# For backward compatibility with the wrapper
if __name__ == "__main__":
    # Test as standalone application
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = ElectrodeAdvisorWidget()
    widget.show()
    sys.exit(app.exec())
