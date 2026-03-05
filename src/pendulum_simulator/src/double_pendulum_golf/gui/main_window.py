"""
Main application window for the Double Pendulum Golf Swing Simulator.

Orchestrates the three sub-widgets (PendulumWidget, MatrixWidget,
ControlsWidget), manages the simulation lifecycle, and drives the
animation timer.
"""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QStatusBar,
    QLabel,
    QTabWidget,
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
        self.resize(1200, 700)
        self.setMinimumSize(900, 550)
        self._set_dark_theme()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _set_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #1a1a24; }
            QStatusBar { background: #1a1a24; color: #9090b0; font-size: 11px; }
            QSplitter::handle { background: #303048; width: 3px; }
            QLabel { color: #c0c0d8; }
        """)

    def _build_ui(self) -> None:
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Title
        title = QLabel(self.WINDOW_TITLE)
        title.setFont(QFont("Sans", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #d0d0e8; padding: 4px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._build_double_panel(), "Double Pendulum")
        tabs.addTab(self._build_triple_panel(), "Triple Pendulum")
        main_layout.addWidget(tabs, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — configure parameters and run simulation")

    def _build_double_panel(self) -> SimulationPanel:
        controls = ControlsWidget()
        pendulum = PendulumWidget()
        matrix = MatrixWidget()
        torque_history = TorqueHistoryWidget()

        def build_params(p: dict) -> PendulumParams:
            return PendulumParams(
                m1=p["m1"],
                m2=p["m2"],
                L1=p["L1"],
                L2=p["L2"],
                b1=p.get("b1", 0.0),
                b2=p.get("b2", 0.0),
                mu1=p.get("mu1", 0.0),
                mu2=p.get("mu2", 0.0),
            )

        def build_state(p: dict) -> np.ndarray:
            return np.array(
                [
                    p["theta1_rad"],
                    p["phi_rad"],
                    p["dtheta1"],
                    p["dphi"],
                ]
            )

        def build_torque(p: dict) -> object:
            return make_polynomial_torque(p["shoulder_coeffs"], p["wrist_coeffs"])

        return SimulationPanel(
            controls=controls,
            pendulum=pendulum,  # type: ignore[arg-type]
            matrix=matrix,  # type: ignore[arg-type]
            params_builder=build_params,
            torque_builder=build_torque,
            state_builder=build_state,
            run_simulation=run_simulation,
            torque_history=torque_history,  # type: ignore[arg-type]
        )

    def _build_triple_panel(self) -> SimulationPanel:
        controls = ControlsWidgetTriple()
        pendulum = PendulumWidget()
        matrix = TripleMatrixWidget()

        def build_params(p: dict) -> TriplePendulumParams:
            return TriplePendulumParams(
                m1=p["m1"],
                m2=p["m2"],
                m3=p["m3"],
                L1=p["L1"],
                L2=p["L2"],
                L3=p["L3"],
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

        return SimulationPanel(
            controls=controls,
            pendulum=pendulum,  # type: ignore[arg-type]
            matrix=matrix,  # type: ignore[arg-type]
            params_builder=build_params,
            torque_builder=build_torque,
            state_builder=build_state,
            run_simulation=run_simulation_triple,
        )
