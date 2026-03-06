"""
Main application window for the Double Pendulum Golf Swing Simulator.

Orchestrates the three sub-widgets (PendulumWidget, MatrixWidget,
ControlsWidget), manages the simulation lifecycle, and drives the
animation timer.

New in UI/UX upgrade:
- QSettings persistence for window geometry
- gravity_on wired to physics g parameter (0.0 when off, 9.81 when on)
- Polished dark chrome styling
"""

import numpy as np
from PyQt6.QtCore import QByteArray, QSettings, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..physics import PendulumParams
from ..physics_triple import TriplePendulumParams
from ..simulation import make_polynomial_torque, run_simulation
from ..simulation_triple import make_polynomial_torque as make_polynomial_torque_triple
from ..simulation_triple import run_simulation as run_simulation_triple
from .controls_widget import ControlsWidget
from .controls_widget_triple import ControlsWidgetTriple
from .matrix_widget import MatrixWidget
from .matrix_widget_triple import TripleMatrixWidget
from .pendulum_widget import PendulumWidget
from .simulation_panel import SimulationPanel
from .torque_history_widget import TorqueHistoryWidget

_SETTINGS_ORG = "D-sorganization"
_SETTINGS_APP = "PendulumSimulator"


class MainWindow(QMainWindow):
    """Top-level window for the double pendulum simulator.

    Layout:
        ┌───────────────────────────────────────────────────┐
        │  Title Bar                                         │
        ├──────────┬──────────────────┬─────────────────────┤
        │ Controls │  Pendulum Canvas │  Mass Matrix Panel   │
        │  (input  │  (animation)     │  (real-time values) │
        │   panel) │                  │                      │
        ├──────────┴──────────────────┴─────────────────────┤
        │  Status Bar                                        │
        └───────────────────────────────────────────────────┘
    """

    WINDOW_TITLE = "Double Pendulum — Golf Swing Dynamics"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1400, 800)
        self.setMinimumSize(900, 550)
        self._set_dark_theme()
        self._build_ui()
        self._restore_geometry()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _set_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #12121c; }
            QStatusBar  { background: #12121c; color: #7878a0; font-size: 11px; border-top: 1px solid #282840; }
            QTabWidget::pane { border: 1px solid #303050; background: #12121c; }
            QTabBar::tab {
                background: #1e1e30; color: #9090b0; border: 1px solid #303050;
                padding: 7px 18px; margin-right: 2px; border-bottom: none;
                font-size: 12px;
            }
            QTabBar::tab:selected { background: #282848; color: #d0d0f0; border-bottom: 2px solid #6070c0; }
            QTabBar::tab:hover    { background: #222238; color: #c0c0e8; }
            QSplitter::handle { background: #282848; width: 4px; }
            QSplitter::handle:hover { background: #404068; }
            QScrollBar:vertical {
                background: #1a1a2a; width: 10px; border: none;
            }
            QScrollBar::handle:vertical {
                background: #404060; min-height: 20px; border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover { background: #5060a0; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: #1a1a2a; height: 10px; border: none;
            }
            QScrollBar::handle:horizontal {
                background: #404060; min-width: 20px; border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover { background: #5060a0; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            QLabel { color: #c0c0d8; }
        """)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(4)

        # Title
        title = QLabel(self.WINDOW_TITLE)
        title.setFont(QFont("Sans", 15, QFont.Weight.Bold))
        title.setStyleSheet("color: #c8c8f0; padding: 4px; letter-spacing: 0.5px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        self._tabs = QTabWidget()
        self._double_panel = self._build_double_panel()
        self._triple_panel = self._build_triple_panel()
        self._tabs.addTab(self._double_panel, "⚙ Double Pendulum")
        self._tabs.addTab(self._triple_panel, "⚙ Triple Pendulum")
        main_layout.addWidget(self._tabs, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Ready  ·  Scroll to zoom  ·  Drag to pan  ·  Double-click to reset view"
        )

    def _build_double_panel(self) -> SimulationPanel:
        controls = ControlsWidget()
        pendulum = PendulumWidget()
        matrix = MatrixWidget()
        torque_history = TorqueHistoryWidget()

        def build_params(p: dict) -> PendulumParams:
            g = 9.81 if p.get("gravity_on", True) else 0.0
            return PendulumParams(
                m1=p["m1"],
                m2=p["m2"],
                L1=p["L1"],
                L2=p["L2"],
                g=g,
                b1=p.get("b1", 0.0),
                b2=p.get("b2", 0.0),
                mu1=p.get("mu1", 0.0),
                mu2=p.get("mu2", 0.0),
            )

        def build_state(p: dict) -> np.ndarray:
            return np.array([p["theta1_rad"], p["phi_rad"], p["dtheta1"], p["dphi"]])

        def build_torque(p: dict) -> object:
            return make_polynomial_torque(p["shoulder_coeffs"], p["wrist_coeffs"])

        panel = SimulationPanel(
            controls=controls,
            pendulum=pendulum,  # type: ignore[arg-type]
            matrix=matrix,  # type: ignore[arg-type]
            params_builder=build_params,
            torque_builder=build_torque,
            state_builder=build_state,
            run_simulation=run_simulation,
            torque_history=torque_history,  # type: ignore[arg-type]
        )
        panel._settings_key = "splitter_double"
        return panel

    def _build_triple_panel(self) -> SimulationPanel:
        controls = ControlsWidgetTriple()
        pendulum = PendulumWidget()
        matrix = TripleMatrixWidget()

        def build_params(p: dict) -> TriplePendulumParams:
            g = 9.81 if p.get("gravity_on", True) else 0.0
            return TriplePendulumParams(
                m1=p["m1"],
                m2=p["m2"],
                m3=p["m3"],
                L1=p["L1"],
                L2=p["L2"],
                L3=p["L3"],
                g=g,
            )

        def build_state(p: dict) -> np.ndarray:
            return np.array(
                [
                    p["theta1_rad"],
                    p["phi1_rad"],
                    p["phi2_rad"],
                    p["dtheta1"],
                    p["dphi1"],
                    p["dphi2"],
                ]
            )

        def build_torque(p: dict) -> object:
            return make_polynomial_torque_triple(
                p["shoulder_coeffs"], p["elbow_coeffs"], p["wrist_coeffs"]
            )

        panel = SimulationPanel(
            controls=controls,
            pendulum=pendulum,  # type: ignore[arg-type]
            matrix=matrix,  # type: ignore[arg-type]
            params_builder=build_params,
            torque_builder=build_torque,
            state_builder=build_state,
            run_simulation=run_simulation_triple,
        )
        panel._settings_key = "splitter_triple"
        return panel

    # ------------------------------------------------------------------
    # Window geometry persistence
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        geom = settings.value("window_geometry")
        if isinstance(geom, QByteArray):
            self.restoreGeometry(geom)

    def closeEvent(self, event: object) -> None:
        """Save window geometry and splitter states on close."""
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("window_geometry", self.saveGeometry())
        # Also persist splitters
        self._double_panel.save_layout()
        self._triple_panel.save_layout()
        super().closeEvent(event)  # type: ignore[arg-type]
