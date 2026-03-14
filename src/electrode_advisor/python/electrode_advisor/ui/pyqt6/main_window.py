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

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any

import matplotlib as mpl

# Set environment variable before any Qt imports
os.environ["QT_API"] = "pyqt6"

# Set matplotlib backend to PyQt6 BEFORE any other imports
if os.environ.get("HEADLESS", "false").lower() == "true":
    with contextlib.suppress(ImportError, ValueError):
        mpl.use("Agg")
else:
    try:
        mpl.use("QtAgg")
    except (ImportError, ValueError):
        mpl.use("Agg")  # Fallback to non-interactive backend

# Prevent matplotlib from opening separate windows
mpl.rcParams["interactive"] = False  # Disable interactive mode


if TYPE_CHECKING:
    pass

from PyQt6.QtCore import Qt, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QCloseEvent  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QWidget,
)
from upstream_drift_tools.calculators.electrical import (  # noqa: E402
    ElectrodeConfig,
    GlassPropertiesInterface,
    ThreePhaseElectricalModelEnhanced,
)

from ...utils.visualization import ElectrodeVisualization  # noqa: E402

logger = logging.getLogger(__name__)

# Existing mixins
# New decomposition mixins
from .main_window_calculation import CalculationMixin  # noqa: E402
from .main_window_drawing import DrawingMixin  # noqa: E402
from .main_window_glass_integration import GlassIntegrationMixin  # noqa: E402
from .main_window_input_panel import InputPanelMixin  # noqa: E402
from .main_window_interaction import InteractionMixin  # noqa: E402
from .main_window_persistence import PersistenceMixin  # noqa: E402
from .main_window_results_charts import ResultsAndChartsMixin  # noqa: E402
from .main_window_state_export import StateAndExportMixin  # noqa: E402
from .main_window_ui_setup import UISetupMixin  # noqa: E402
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
    UISetupMixin,
    PersistenceMixin,
    CalculationMixin,
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
            vertical_spreading: Initial vertical spreading factor
            horizontal_spreading: Initial horizontal spreading factor
        """
        super().__init__(parent)

        self.calculator_name = calculator_name
        self.config = config or ElectrodeConfig()
        self.glass_interface = glass_interface
        self.electrical_model = ThreePhaseElectricalModelEnhanced()
        self.visualization = ElectrodeVisualization(self.config)

        # Store spreading factors
        self.config.vertical_spreading_factor = vertical_spreading
        self.config.horizontal_spreading_factor = horizontal_spreading

        # State management variables
        self.calculation_results: dict[str, Any] = {}
        self._initialization_complete = False

        # Depth inputs dictionary
        self.depth_inputs: dict[int, Any] = {}
        self.phase_inputs: dict[str, dict[str, Any]] = {}

        # Initialize UI (from UISetupMixin)
        self._init_ui()

        # Connect checkbox signals (from UISetupMixin)
        self._connect_checkbox_signals()

        # Setup timers (from CalculationMixin)
        self._setup_timers()

        # Setup state management (from PersistenceMixin)
        self.setup_state_management()

        # Load previous state
        self.load_state()

        # Mark initialization as complete
        self._initialization_complete = True

        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    # Input panel methods moved to InputPanelMixin (main_window_input_panel.py)
    # Visual controls methods moved to VisualControlsMixin (main_window_visual_controls.py)
    # Results/charts update methods moved to ResultsAndChartsMixin (main_window_results_charts.py)
    # Mouse/interaction methods -> InteractionMixin (main_window_interaction.py)
    # View presets/color scheme -> ViewPresetsMixin (main_window_view_presets.py)
    # State/export methods -> StateAndExportMixin (main_window_state_export.py)
    # UI setup methods -> UISetupMixin (main_window_ui_setup.py)
    # Persistence methods -> PersistenceMixin (main_window_persistence.py)
    # Calculation methods -> CalculationMixin (main_window_calculation.py)


# For backward compatibility with the wrapper
if __name__ == "__main__":
    # Test as standalone application
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = ElectrodeAdvisorWidget()
    widget.show()
    sys.exit(app.exec())
