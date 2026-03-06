"""
Main application window for the Double Pendulum Golf Swing Simulator.

Orchestrates sub-widgets, manages simulation lifecycle, drives animation.

New in UI/UX upgrade:
- QSettings persistence for window geometry + splitters
- Gravity toggle wired to g=0/9.81 in params builders
- Menu bar: View → Themes (fleet ThemeManager) + quick-switch submenu
- Current dark style preserved as "Pendulum Dark" fallback
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QByteArray, QSettings
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow,
    QMenu,
    QMenuBar,
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
from .toolstrip_widget import ToolStrip
from .torque_history_widget import TorqueHistoryWidget

logger = logging.getLogger(__name__)

_SETTINGS_ORG = "D-sorganization"
_SETTINGS_APP = "PendulumSimulator"

# ── Try to import fleet ThemeManager ─────────────────────────────────────────
_THEME_AVAILABLE = False
ThemeManager: Any = None
ThemeManagerDialog: Any = None
create_theme_menu: Any = None
try:
    _shared_root = Path(__file__).parents[7] / "shared" / "python"
    if str(_shared_root) not in sys.path and _shared_root.exists():
        sys.path.insert(0, str(_shared_root))
    from theme import (  # type: ignore[no-redef]
        ThemeManager,
        ThemeManagerDialog,
        create_theme_menu,
    )

    _THEME_AVAILABLE = True
except ImportError:
    pass  # ThemeManager / ThemeManagerDialog / create_theme_menu remain None

# ── Pendulum dark stylesheet (preserved regardless of theme system) ────────────
_PENDULUM_DARK_STYLE = """
    QMainWindow { background: #12121c; }
    QStatusBar  { background: #12121c; color: #7878a0; font-size: 11px;
                  border-top: 1px solid #282840; }
    QTabWidget::pane { border: 1px solid #303050; background: #12121c; }
    QTabBar::tab { background: #1e1e30; color: #9090b0; border: 1px solid #303050;
                   padding: 7px 18px; margin-right: 2px; border-bottom: none;
                   font-size: 12px; }
    QTabBar::tab:selected { background: #282848; color: #d0d0f0;
                            border-bottom: 2px solid #6070c0; }
    QTabBar::tab:hover    { background: #222238; color: #c0c0e8; }
    QSplitter::handle { background: #282848; width: 4px; }
    QSplitter::handle:hover { background: #404068; }
    QScrollBar:vertical { background: #1a1a2a; width: 10px; border: none; }
    QScrollBar::handle:vertical { background: #404060; min-height: 20px;
                                  border-radius: 5px; }
    QScrollBar::handle:vertical:hover { background: #5060a0; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar:horizontal { background: #1a1a2a; height: 10px; border: none; }
    QScrollBar::handle:horizontal { background: #404060; min-width: 20px;
                                    border-radius: 5px; }
    QScrollBar::handle:horizontal:hover { background: #5060a0; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
    QLabel { color: #c0c0d8; }
    QMenuBar { background: #16162a; color: #b0b0d0; font-size: 11px; }
    QMenuBar::item:selected { background: #282848; }
    QMenu { background: #1e1e30; color: #c0c0d8; border: 1px solid #404060;
            font-size: 11px; }
    QMenu::item:selected { background: #383868; }
"""


class MainWindow(QMainWindow):
    """Top-level window for the double pendulum simulator."""

    WINDOW_TITLE = "Pendulums"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1400, 800)
        self.setMinimumSize(900, 550)

        # Apply base dark style (always)
        self.setStyleSheet(_PENDULUM_DARK_STYLE)

        # Set app favicon
        _icon_path = Path(__file__).parent / "pendulum_icon.png"
        if _icon_path.exists():
            from PyQt6.QtGui import QIcon

            self.setWindowIcon(QIcon(str(_icon_path)))

        self._theme_manager: object | None = None
        self._build_menu()
        self._build_ui()
        self._setup_theme()
        self._restore_geometry()

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        _mb = self.menuBar()
        assert _mb is not None
        menubar: QMenuBar = _mb

        # View menu
        _view = menubar.addMenu("&View")
        assert _view is not None
        view_menu: QMenu = _view

        # Quick theme submenu
        self._quick_theme_menu = view_menu.addMenu("🎨 Quick Theme")

        # Full theme manager action
        self._action_theme_mgr = QAction("Theme Manager…", self)
        self._action_theme_mgr.setShortcut("Ctrl+Shift+T")
        self._action_theme_mgr.triggered.connect(self._open_theme_manager)
        view_menu.addAction(self._action_theme_mgr)

        view_menu.addSeparator()

        # Always-available "Pendulum Dark" built-in
        action_pend_dark = QAction("Pendulum Dark (default)", self)
        action_pend_dark.triggered.connect(self._apply_pendulum_dark)
        view_menu.addAction(action_pend_dark)

        # Help menu
        _help = menubar.addMenu("&Help")
        assert _help is not None
        action_about = QAction("About…", self)
        action_about.triggered.connect(self._show_about)
        _help.addAction(action_about)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Persistent toolstrip — always visible, regardless of scroll
        self._toolstrip = ToolStrip()
        main_layout.addWidget(self._toolstrip)

        self._tabs = QTabWidget()
        self._double_panel = self._build_double_panel()
        self._triple_panel = self._build_triple_panel()
        self._tabs.addTab(self._double_panel, "⚙ Double Pendulum")
        self._tabs.addTab(self._triple_panel, "⚙ Triple Pendulum")
        main_layout.addWidget(self._tabs, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Ready  ·  Scroll=zoom  ·  Drag=pan  ·  Dbl-click=reset view",
        )

        self._wire_toolstrip()

    def _wire_toolstrip(self) -> None:
        """Connect toolstrip signals to both pendulum panels."""
        from typing import cast

        ts = self._toolstrip

        # Only forward Run/Reset/Play/Speed to the ACTIVE tab's panel
        # (both panels subscribe; the hidden one just ignores it)
        for panel in (self._double_panel, self._triple_panel):
            pw = cast(PendulumWidget, panel.pendulum)

            # Simulation actions
            ts.run_requested.connect(panel.controls.run_requested.emit)
            ts.reset_requested.connect(panel.controls.reset_requested.emit)
            ts.play_toggled.connect(panel.controls.play_toggled.emit)
            ts.speed_changed.connect(panel.controls.speed_changed.emit)

            # Playback scrubbing via toolstrip slider
            ts.frame_scrubbed.connect(panel.scrub_to_frame)

            # Overlay toggles → pendulum widget
            ts.forces_toggled.connect(pw.set_show_forces)
            ts.mob_ellipsoid_toggled.connect(pw.set_show_mob_ellipsoids)
            ts.force_ellipsoid_toggled.connect(pw.set_show_force_ellipsoids)

            # Scale sliders → pendulum widget
            ts.force_scale_changed.connect(pw.set_force_scale)
            ts.mob_scale_changed.connect(pw.set_mob_ellipsoid_scale)
            ts.force_ell_scale_changed.connect(pw.set_force_ellipsoid_scale)

            # Busy state
            panel.sim_started.connect(lambda: ts.set_running(True))
            panel.sim_finished.connect(lambda: ts.set_running(False))

            # Sync toolstrip slider when simulation completes
            panel.sim_finished.connect(
                lambda p=panel: ts.set_frame_range(p.current_n_steps())
            )

            # Sync toolstrip frame counter when animation advances
            panel.frame_changed.connect(lambda idx: ts.set_frame(idx))

            # Reset view button → clear zoom/pan on the pendulum canvas
            ts.reset_view_requested.connect(pw.reset_view)

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
                ],
            )

        def build_torque(p: dict) -> object:
            return make_polynomial_torque_triple(
                p["shoulder_coeffs"],
                p["elbow_coeffs"],
                p["wrist_coeffs"],
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
    # Theme management
    # ------------------------------------------------------------------

    def _setup_theme(self) -> None:
        """Wire fleet ThemeManager if available; populate quick-theme menu."""
        if not _THEME_AVAILABLE or ThemeManager is None:
            logger.info("theme package unavailable — using Pendulum Dark built-in")
            return

        try:
            self._theme_manager = ThemeManager.instance(
                main_window=self,
                app_context="PendulumSimulator",
                settings_org=_SETTINGS_ORG,
                settings_app=_SETTINGS_APP,
            )
            # Apply saved theme
            self._theme_manager.apply_theme()  # type: ignore[union-attr]
            self._theme_manager.themeChanged.connect(self._on_theme_changed)  # type: ignore[union-attr]

            # Use shared helper to build a full theme submenu (window first, then parent)
            assert self._quick_theme_menu is not None
            if create_theme_menu is not None:
                create_theme_menu(
                    self,
                    self._quick_theme_menu,
                    show_custom_options=True,
                )

        except Exception:
            logger.exception("Failed to initialise ThemeManager")
            self._theme_manager = None

    def _on_theme_changed(self, name: str) -> None:
        self.status.showMessage(f"Theme changed to: {name}", 3000)

    def _open_theme_manager(self) -> None:
        if (
            not _THEME_AVAILABLE
            or self._theme_manager is None
            or ThemeManagerDialog is None
        ):
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Themes",
                "The fleet theme package is not installed.\n\n"
                "Use View → Pendulum Dark to reset to the default style.",
            )
            return
        dlg = ThemeManagerDialog(self._theme_manager, self)
        dlg.exec()

    def _apply_pendulum_dark(self) -> None:
        """Force-reset to the built-in pendulum dark stylesheet."""
        self.setStyleSheet(_PENDULUM_DARK_STYLE)
        self.status.showMessage("Theme: Pendulum Dark", 3000)

    def _show_about(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About",
            "<b>Double Pendulum Golf Swing Simulator</b><br><br>"
            "Interactive simulation of 2- and 3-segment pendulum dynamics.<br><br>"
            "Built with PyQt6 · NumPy · SciPy<br>"
            "D-sorganization Tools Repository",
        )

    # ------------------------------------------------------------------
    # Geometry persistence
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        geom = settings.value("window_geometry")
        if isinstance(geom, QByteArray):
            self.restoreGeometry(geom)

    def closeEvent(self, event: object) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue("window_geometry", self.saveGeometry())
        self._double_panel.save_layout()
        self._triple_panel.save_layout()
        super().closeEvent(event)  # type: ignore[arg-type]
