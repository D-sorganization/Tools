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
from collections.abc import Callable
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

from ..physics import JointLimits, PendulumParams, TorqueClamp
from ..physics_golfer import GolferParams
from ..physics_triple import TriplePendulumParams
from ..simulation import make_polynomial_torque, run_simulation
from ..simulation_golfer import (
    make_polynomial_torque as make_polynomial_torque_golfer,
)
from ..simulation_golfer import run_simulation as run_simulation_golfer
from ..simulation_triple import (
    make_polynomial_torque as make_polynomial_torque_triple,
)
from ..simulation_triple import run_simulation as run_simulation_triple
from .controls_widget import ControlsWidget
from .controls_widget_golfer import ControlsWidgetGolfer
from .controls_widget_triple import ControlsWidgetTriple
from .golfer_pendulum_widget import GolferPendulumWidget
from .matrix_widget import MatrixWidget
from .matrix_widget_golfer import GolferMatrixWidget
from .matrix_widget_triple import TripleMatrixWidget
from .pendulum_widget import PendulumWidget
from .simulation_panel import SimulationPanel
from .toolstrip_widget import ToolStrip
from .torque_history_widget import TorqueHistoryWidget
from .optimization_widget import OptimizationWidget

# TODO(#1042): Derive from fleet ThemeManager palette when it's a hard dep.
from .controls_utils import PENDULUM_DARK_STYLE as _PENDULUM_DARK_STYLE

logger = logging.getLogger(__name__)

_SETTINGS_ORG = "D-sorganization"
_SETTINGS_APP = "PendulumSimulator"

# ── Try to import fleet ThemeManager ─────────────────────────────────────────
_THEME_AVAILABLE = False
ThemeManager: Any = None
ThemeManagerDialog: Any = None
create_theme_menu: Any = None
try:
    # Walk up from this file to find the 'shared/python' directory
    _shared_root = None
    _p = Path(__file__).resolve().parent
    for _ in range(10):
        _candidate = _p / "shared" / "python"
        if _candidate.exists():
            _shared_root = _candidate
            break
        _p = _p.parent

    if _shared_root is not None and str(_shared_root) not in sys.path:
        sys.path.insert(0, str(_shared_root))
    from theme import (
        ThemeManager as _ThemeManager,
        ThemeManagerDialog as _ThemeManagerDialog,
        create_theme_menu as _create_theme_menu,
    )

    ThemeManager = _ThemeManager
    ThemeManagerDialog = _ThemeManagerDialog
    create_theme_menu = _create_theme_menu
    _THEME_AVAILABLE = True
except ImportError:
    pass  # ThemeManager / ThemeManagerDialog / create_theme_menu remain None

# ── Try to import shared PlotThemeManager ──────────────────────────────────
_PLOT_THEME_AVAILABLE = False
create_plot_theme_menu: Any = None
try:
    from plot_theme.integration import (
        create_plot_theme_menu as _shared_create_plot_theme_menu,
    )

    create_plot_theme_menu = _shared_create_plot_theme_menu
    _PLOT_THEME_AVAILABLE = True
except ImportError:
    pass


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

        # Plot Theme submenu (for pyqtgraph / matplotlib colours)
        if _PLOT_THEME_AVAILABLE and create_plot_theme_menu is not None:
            view_menu.addSeparator()
            create_plot_theme_menu(self, menubar)

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
        self._golfer_panel = self._build_golfer_panel()
        self._tabs.addTab(self._double_panel, "⚙ Double Pendulum")
        self._tabs.addTab(self._triple_panel, "⚙ Triple Pendulum")
        self._tabs.addTab(self._golfer_panel, "⚙ Golfer Upper Body")
        main_layout.addWidget(self._tabs, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Ready  ·  Scroll=zoom  ·  Drag=pan  ·  Dbl-click=reset view",
        )

        self._wire_toolstrip()

    def _wire_toolstrip(self) -> None:
        """Connect toolstrip signals to both pendulum panels."""
        ts = self._toolstrip

        # Only forward Run/Reset/Play/Speed to the ACTIVE tab's panel
        # (both panels subscribe; the hidden one just ignores it)
        all_panels = (
            self._double_panel,
            self._triple_panel,
            self._golfer_panel,
        )
        for panel in all_panels:
            pw = panel.pendulum

            # Simulation actions
            ts.run_requested.connect(panel.controls.run_requested.emit)
            ts.reset_requested.connect(panel.controls.reset_requested.emit)
            ts.play_toggled.connect(panel.controls.play_toggled.emit)
            ts.speed_changed.connect(panel.controls.speed_changed.emit)

            # Playback scrubbing via toolstrip slider
            ts.frame_scrubbed.connect(panel.scrub_to_frame)

            # Overlay toggles → pendulum widget (optional capability)
            if hasattr(pw, "set_show_forces"):
                ts.forces_toggled.connect(pw.set_show_forces)
            if hasattr(pw, "set_show_zero_torque_forces"):
                ts.zero_torque_toggled.connect(pw.set_show_zero_torque_forces)
            if hasattr(pw, "set_show_mob_ellipsoids"):
                ts.mob_ellipsoid_toggled.connect(pw.set_show_mob_ellipsoids)
            if hasattr(pw, "set_show_force_ellipsoids"):
                ts.force_ellipsoid_toggled.connect(pw.set_show_force_ellipsoids)
            if hasattr(pw, "set_show_com"):
                ts.com_toggled.connect(pw.set_show_com)

            # Scale sliders → pendulum widget (optional capability)
            if hasattr(pw, "set_force_scale"):
                ts.force_scale_changed.connect(pw.set_force_scale)
            if hasattr(pw, "set_mob_ellipsoid_scale"):
                ts.mob_scale_changed.connect(pw.set_mob_ellipsoid_scale)
            if hasattr(pw, "set_force_ellipsoid_scale"):
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
            if hasattr(pw, "reset_view"):
                ts.reset_view_requested.connect(pw.reset_view)

            # Per-segment overlay visibility (#1100, #1101, #1102)
            if hasattr(pw, "set_visible_segments"):
                ts.segment_visibility_changed.connect(pw.set_visible_segments)

        # Update segment checkboxes when tab changes
        self._tabs.currentChanged.connect(self._on_tab_changed)
        # Initialize with the default tab's segments
        self._on_tab_changed(self._tabs.currentIndex())

    def _build_double_panel(self) -> SimulationPanel:
        controls = ControlsWidget()
        pendulum = PendulumWidget()
        matrix = MatrixWidget()
        torque_history = TorqueHistoryWidget()

        def build_params(p: dict) -> PendulumParams:
            tilt_rad = np.radians(p.get("tilt_deg", 0.0))
            azimuth_rad = np.radians(p.get("azimuth_deg", 0.0))
            g = 9.81 if p.get("gravity_on", True) else 0.0
            g_eff = g * float(np.cos(tilt_rad))  # (#1113) effective gravity on plane
            # Update display tilt and view azimuth on next paint
            pendulum.set_tilt_angle(tilt_rad)
            pendulum.set_view_azimuth(azimuth_rad)  # (#1118)
            return PendulumParams(
                m1=p["m1"],
                m2=p["m2"],
                L1=p["L1"],
                L2=p["L2"],
                mClub=p.get("mClub", 0.0),
                g=g_eff,
                b1=p.get("b1", 0.0),
                b2=p.get("b2", 0.0),
                mu1=p.get("mu1", 0.0),
                mu2=p.get("mu2", 0.0),
            )

        def build_state(p: dict) -> np.ndarray:
            return np.array([p["theta1_rad"], p["phi_rad"], p["dtheta1"], p["dphi"]])

        def build_torque(p: dict) -> object:
            return make_polynomial_torque(p["shoulder_coeffs"], p["wrist_coeffs"])

        def build_limits(p: dict) -> JointLimits | None:
            if not p.get("enable_limits", False):
                return None
            return JointLimits(
                phi_min=p.get("phi_min_rad", -np.pi / 2),
                phi_max=p.get("phi_max_rad", np.pi / 2),
                stiffness=p.get("limit_stiffness", 500.0),
            )

        def build_clamp(p: dict) -> TorqueClamp | None:
            if not p.get("enable_clamp", False):
                return None
            return TorqueClamp(
                max_torque1=p.get("max_torque1", 50.0),
                max_torque2=p.get("max_torque2", 20.0),
            )

        # Optimizer (#1108)
        optimizer = OptimizationWidget(
            model_name="Double Pendulum",
            n_torque_params=2,
        )

        def _make_double_objective(p: dict) -> Callable:
            """Build a tip-speed objective from current controls."""
            params = build_params(p)
            initial_state = build_state(p)
            t_end = p["t_end"]
            limits = build_limits(p)
            clamp = build_clamp(p)

            def objective(coeffs: np.ndarray) -> float:
                n_half = len(coeffs) // 2
                s_coeffs = list(coeffs[:n_half])
                w_coeffs = list(coeffs[n_half:])
                torque_func = make_polynomial_torque(s_coeffs, w_coeffs)
                try:
                    result = run_simulation(
                        params=params,
                        initial_state=initial_state,
                        t_end=t_end,
                        torque_func=torque_func,
                        limits=limits,
                        clamp=clamp,
                    )
                    # Tip speed at last frame
                    vels = result.joint_velocities_at(result.n_steps - 1)
                    tip_v = vels.get("tip", (0, 0))
                    speed = float(np.hypot(tip_v[0], tip_v[1]))
                    return -speed  # minimize negative speed
                except Exception:  # noqa: BLE001
                    return 0.0  # crashed → bad solution

            return objective

        panel = SimulationPanel(
            controls=controls,
            pendulum=pendulum,  # type: ignore[arg-type]
            matrix=matrix,  # type: ignore[arg-type]
            params_builder=build_params,
            torque_builder=build_torque,
            state_builder=build_state,
            run_simulation=run_simulation,
            torque_history=torque_history,  # type: ignore[arg-type]
            limits_builder=build_limits,
            clamp_builder=build_clamp,
            optimizer=optimizer,
        )
        panel._settings_key = "splitter_double"
        return panel

    def _build_triple_panel(self) -> SimulationPanel:
        controls = ControlsWidgetTriple()
        pendulum = PendulumWidget()
        matrix = TripleMatrixWidget()

        def build_params(p: dict) -> TriplePendulumParams:
            tilt_rad = np.radians(p.get("tilt_deg", 0.0))
            g = 9.81 if p.get("gravity_on", True) else 0.0
            g_eff = g * float(np.cos(tilt_rad))  # (#1113)
            pendulum.set_tilt_angle(tilt_rad)
            pendulum.set_view_azimuth(np.radians(p.get("azimuth_deg", 0.0)))  # (#1118)
            return TriplePendulumParams(
                m1=p["m1"],
                m2=p["m2"],
                m3=p["m3"],
                L1=p["L1"],
                L2=p["L2"],
                L3=p["L3"],
                g=g_eff,
                b1=p.get("b1", 0.0),
                b2=p.get("b2", 0.0),
                b3=p.get("b3", 0.0),
                mu1=p.get("mu1", 0.0),
                mu2=p.get("mu2", 0.0),
                mu3=p.get("mu3", 0.0),
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

        # Optimizer (#1109)
        optimizer = OptimizationWidget(
            model_name="Triple Pendulum",
            n_torque_params=3,
        )

        panel = SimulationPanel(
            controls=controls,
            pendulum=pendulum,  # type: ignore[arg-type]
            matrix=matrix,  # type: ignore[arg-type]
            params_builder=build_params,
            torque_builder=build_torque,
            state_builder=build_state,
            run_simulation=run_simulation_triple,
            optimizer=optimizer,
        )
        panel._settings_key = "splitter_triple"
        return panel

    def _build_golfer_panel(self) -> SimulationPanel:
        controls = ControlsWidgetGolfer()
        pendulum = GolferPendulumWidget()
        matrix = GolferMatrixWidget()

        def build_params(p: dict) -> GolferParams:
            tilt_rad = np.radians(p.get("tilt_deg", 0.0))
            g = 9.81 if p.get("gravity_on", True) else 0.0
            g_eff = g * float(np.cos(tilt_rad))  # (#1113)
            pendulum.set_tilt_angle(tilt_rad)
            pendulum.set_view_azimuth(np.radians(p.get("azimuth_deg", 0.0)))  # (#1118)
            return GolferParams(
                m_hub=p["m_hub"],
                m_r_upper=p["m_r_upper"],
                m_r_fore=p["m_r_fore"],
                m_l_upper=p["m_l_upper"],
                m_l_fore=p["m_l_fore"],
                m_club=p["m_club"],
                L_hub=p["L_hub"],
                L_r_upper=p["L_r_upper"],
                L_r_fore=p["L_r_fore"],
                L_l_upper=p["L_l_upper"],
                L_l_fore=p["L_l_fore"],
                L_club=p["L_club"],
                d_rs=p["d_rs"],
                d_ls=p["d_ls"],
                grip_right=p["grip_right"],
                grip_left=p["grip_left"],
                m_clubhead=p.get("m_clubhead", 0.2),
                g=g_eff,
                b_hub=p.get("b_hub", 0.0),
                b_rs=p.get("b_rs", 0.0),
                b_re=p.get("b_re", 0.0),
                b_rh=p.get("b_rh", 0.0),
                b_ls=p.get("b_ls", 0.0),
                b_le=p.get("b_le", 0.0),
                b_lh=p.get("b_lh", 0.0),
            )

        def build_state(p: dict) -> np.ndarray:
            return np.array(
                [
                    p["theta_hub_rad"],
                    p["alpha_rs_rad"],
                    p["alpha_re_rad"],
                    p["alpha_rh_rad"],
                    p["alpha_ls_rad"],
                    p["alpha_le_rad"],
                    p["alpha_lh_rad"],
                    0.0,  # theta_club (computed by projection)
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # qdot (all zero)
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )

        def build_torque(p: dict) -> object:
            return make_polynomial_torque_golfer(
                p["hub_coeffs"],
                p["rs_coeffs"],
                p["re_coeffs"],
                p["rh_coeffs"],
                p["ls_coeffs"],
                p["le_coeffs"],
                p["lh_coeffs"],
            )

        # Optimizer (#1110)
        optimizer = OptimizationWidget(
            model_name="Golfer Upper Body",
            n_torque_params=7,
        )

        panel = SimulationPanel(
            controls=controls,
            pendulum=pendulum,  # type: ignore[arg-type]
            matrix=matrix,  # type: ignore[arg-type]
            params_builder=build_params,
            torque_builder=build_torque,
            state_builder=build_state,
            run_simulation=run_simulation_golfer,
            optimizer=optimizer,
        )
        panel._settings_key = "splitter_golfer"
        return panel

    # ------------------------------------------------------------------
    # Per-segment visibility (#1100, #1101, #1102)
    # ------------------------------------------------------------------

    # Joint names per model type
    _SEGMENTS_DOUBLE = ["shoulder", "wrist", "tip"]
    _SEGMENTS_TRIPLE = ["shoulder", "wrist1", "wrist2", "tip"]
    _SEGMENTS_GOLFER = ["hub", "rs", "re", "rh", "ls", "le", "lh", "club_tip"]

    def _on_tab_changed(self, index: int) -> None:
        """Update toolstrip segment checkboxes for the active model tab."""
        segment_map = {
            0: self._SEGMENTS_DOUBLE,
            1: self._SEGMENTS_TRIPLE,
            2: self._SEGMENTS_GOLFER,
        }
        names = segment_map.get(index, self._SEGMENTS_DOUBLE)
        self._toolstrip.set_segment_names(names)

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
            "Interactive simulation of 2-, 3-segment, and golfer"
            " upper-body pendulum dynamics.<br><br>"
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
        self._golfer_panel.save_layout()
        super().closeEvent(event)  # type: ignore[arg-type]
