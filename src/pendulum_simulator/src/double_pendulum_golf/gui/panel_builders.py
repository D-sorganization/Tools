"""
Panel builder functions extracted from MainWindow.

Each builder creates and wires up a complete simulation panel for
a specific pendulum model (double, triple, golfer).

Design by Contract
------------------
- build_double_panel(main_window) returns a fully wired SimulationPanel.
- build_triple_panel(main_window) returns a fully wired SimulationPanel.
- build_golfer_panel(main_window) returns a fully wired SimulationPanel.
- wire_toolstrip(main_window) connects toolstrip signals.

DRY
---
Common panel setup logic is factored into _connect_common_signals().
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from ..physics import (
    JointLimits,
    JointLimitsNDOF,
    PendulumParams,
    TorqueClamp,
)
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
from .torque_history_widget import TorqueHistoryWidget
from .optimization_widget import OptimizationWidget

logger = logging.getLogger(__name__)


def build_double_panel(main_window: Any) -> SimulationPanel:
    """Build and return the double pendulum simulation panel.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance (used to access state if needed).

    Returns
    -------
    SimulationPanel
        A fully wired simulation panel for the double pendulum model.
    """
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
            theta1_min=p.get("theta1_min_rad", -np.pi),
            theta1_max=p.get("theta1_max_rad", np.pi),
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
                    torque_func=torque_func,  # type: ignore[arg-type]
                    limits=limits,
                    clamp=clamp,
                )
                # Tip speed at last frame
                vels = result.joint_velocities_at(result.n_steps - 1)
                tip_v = vels.get("tip", (0, 0))
                speed = float(np.hypot(tip_v[0], tip_v[1]))
                return -speed  # minimize negative speed
            except (
                RuntimeError,
                ValueError,
                ArithmeticError,
            ) as exc:  # noqa: BLE001
                logger.debug("double objective simulation failed: %s", exc)
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
        torque_history=torque_history,
        limits_builder=build_limits,
        clamp_builder=build_clamp,
        optimizer=optimizer,
        objective_builder=_make_double_objective,
    )
    panel._settings_key = "splitter_double"
    return panel


def build_triple_panel(main_window: Any) -> SimulationPanel:
    """Build and return the triple pendulum simulation panel.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance (used to access state if needed).

    Returns
    -------
    SimulationPanel
        A fully wired simulation panel for the triple pendulum model.
    """
    controls = ControlsWidgetTriple()
    pendulum = PendulumWidget()
    matrix = TripleMatrixWidget()
    torque_history = TorqueHistoryWidget()

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
            scapula_offset_rad=np.radians(p.get("scapula_deg", 0.0)),
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

    def build_limits(p: dict) -> JointLimitsNDOF | None:
        if not p.get("enable_limits", False):
            return None
        return JointLimitsNDOF(
            angle_min=np.array(p["limit_mins_rad"]),
            angle_max=np.array(p["limit_maxs_rad"]),
            stiffness=p.get("limit_stiffness", 500.0),
        )

    def build_clamp(p: dict) -> np.ndarray | None:
        if not p.get("enable_clamp", False):
            return None
        return np.array(p["torque_limits"])

    # Optimizer (#1109)
    optimizer = OptimizationWidget(
        model_name="Triple Pendulum",
        n_torque_params=3,
    )

    def _make_triple_objective(p: dict) -> Callable:
        """Build a tip-speed objective from current controls."""
        params = build_params(p)
        initial_state = build_state(p)
        t_end = p["t_end"]
        limits = build_limits(p)
        clamp = build_clamp(p)

        def objective(coeffs: np.ndarray) -> float:
            n_third = len(coeffs) // 3
            s_c = list(coeffs[:n_third])
            e_c = list(coeffs[n_third : 2 * n_third])
            w_c = list(coeffs[2 * n_third :])
            torque_func = make_polynomial_torque_triple(s_c, e_c, w_c)
            try:
                result = run_simulation_triple(
                    params=params,
                    initial_state=initial_state,
                    t_end=t_end,
                    torque_func=torque_func,  # type: ignore[arg-type]
                    torque_limits=clamp,
                    limits=limits,
                )
                vels = result.joint_velocities_at(result.n_steps - 1)  # type: ignore[attr-defined]
                tip_v = vels.get("tip", (0, 0))
                speed = float(np.hypot(tip_v[0], tip_v[1]))
                return -speed
            except (
                RuntimeError,
                ValueError,
                ArithmeticError,
            ) as exc:  # noqa: BLE001
                logger.debug("triple objective simulation failed: %s", exc)
                return 0.0

        return objective

    panel = SimulationPanel(
        controls=controls,
        pendulum=pendulum,  # type: ignore[arg-type]
        matrix=matrix,  # type: ignore[arg-type]
        params_builder=build_params,
        torque_builder=build_torque,
        state_builder=build_state,
        run_simulation=run_simulation_triple,
        torque_history=torque_history,
        limits_builder=build_limits,
        clamp_builder=build_clamp,
        optimizer=optimizer,
        objective_builder=_make_triple_objective,
    )
    panel._settings_key = "splitter_triple"
    return panel


def build_golfer_panel(main_window: Any) -> SimulationPanel:
    """Build and return the golfer upper body simulation panel.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance (used to access state if needed).

    Returns
    -------
    SimulationPanel
        A fully wired simulation panel for the golfer upper body model.
    """
    controls = ControlsWidgetGolfer()
    pendulum = GolferPendulumWidget()
    matrix = GolferMatrixWidget()
    torque_history = TorqueHistoryWidget()

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
            L_rscap=p.get("L_rscap", 0.12),
            L_lscap=p.get("L_lscap", 0.12),
            m_rscap=p.get("m_rscap", 0.5),
            m_lscap=p.get("m_lscap", 0.5),
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

    def build_limits(p: dict) -> JointLimitsNDOF | None:
        if not p.get("enable_limits", False):
            return None
        return JointLimitsNDOF(
            angle_min=np.array(p["limit_mins_rad"]),
            angle_max=np.array(p["limit_maxs_rad"]),
            stiffness=p.get("limit_stiffness", 500.0),
        )

    def build_clamp(p: dict) -> np.ndarray | None:
        if not p.get("enable_clamp", False):
            return None
        return np.array(p["torque_limits"])

    # Optimizer (#1110)
    optimizer = OptimizationWidget(
        model_name="Golfer Upper Body",
        n_torque_params=7,
    )

    def _make_golfer_objective(p: dict) -> Callable:
        """Build a clubhead-speed objective from current controls."""
        params = build_params(p)
        initial_state = build_state(p)
        t_end = p["t_end"]
        limits = build_limits(p)
        clamp = build_clamp(p)

        def objective(coeffs: np.ndarray) -> float:
            n_seventh = max(1, len(coeffs) // 7)
            slices = [
                list(coeffs[i * n_seventh : (i + 1) * n_seventh]) for i in range(7)
            ]
            torque_func = make_polynomial_torque_golfer(*slices)
            try:
                result = run_simulation_golfer(
                    params=params,
                    initial_state=initial_state,
                    t_end=t_end,
                    torque_func=torque_func,  # type: ignore[arg-type]
                    torque_limits=clamp,
                    limits=limits,
                )
                vels = result.joint_velocities_at(result.n_steps - 1)  # type: ignore[attr-defined]
                tip_v = vels.get("club_tip", (0, 0))
                speed = float(np.hypot(tip_v[0], tip_v[1]))
                return -speed
            except (
                RuntimeError,
                ValueError,
                ArithmeticError,
            ) as exc:  # noqa: BLE001
                logger.debug("golfer objective simulation failed: %s", exc)
                return 0.0

        return objective

    panel = SimulationPanel(
        controls=controls,
        pendulum=pendulum,  # type: ignore[arg-type]
        matrix=matrix,  # type: ignore[arg-type]
        params_builder=build_params,
        torque_builder=build_torque,
        state_builder=build_state,
        run_simulation=run_simulation_golfer,
        torque_history=torque_history,
        limits_builder=build_limits,
        clamp_builder=build_clamp,
        optimizer=optimizer,
        objective_builder=_make_golfer_objective,
    )
    panel._settings_key = "splitter_golfer"
    return panel


def wire_toolstrip(main_window: Any) -> None:
    """Connect toolstrip signals — dispatched only to the active tab's panel.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance containing the toolstrip and panels.

    Design by Contract
    ------------------
    Pre: main_window._toolstrip, main_window._tabs, and all panels are initialized.
    Post: All toolstrip signals are wired to their respective handlers.
    """
    ts = main_window._toolstrip

    # Build the ordered panel list matching tab indices
    main_window._panels = (
        main_window._double_panel,
        main_window._triple_panel,
        main_window._golfer_panel,
    )

    # ── Simulation action signals → active panel only ──────────────
    ts.run_requested.connect(
        lambda: main_window._active_panel().controls.run_requested.emit()
    )
    ts.reset_requested.connect(
        lambda: main_window._active_panel().controls.reset_requested.emit()
    )
    ts.play_toggled.connect(
        lambda checked: main_window._active_panel().controls.play_toggled.emit(checked)
    )
    ts.speed_changed.connect(
        lambda val: main_window._active_panel().controls.speed_changed.emit(val)
    )
    ts.frame_scrubbed.connect(
        lambda idx: main_window._active_panel().scrub_to_frame(idx)
    )

    # ── Export actions (#1141) → active panel's controls ──────────
    ts.export_data_requested.connect(
        lambda: main_window._active_panel().controls.export_data_requested.emit()
    )
    ts.export_video_requested.connect(
        lambda: main_window._active_panel().controls.export_video_requested.emit()
    )

    # ── Pop-out chart (#1135) → active panel ─────────────────────
    ts.popout_chart_requested.connect(main_window._on_popout_chart)

    # ── Overlay toggles → active panel's pendulum widget ──────────
    _connect_common_signals(main_window)

    # ── Gravity toggle (#1142) → active panel's pendulum + controls ──
    def _fwd_gravity(on: bool) -> None:
        pw = main_window._active_panel().pendulum
        if hasattr(pw, "set_gravity_on"):
            pw.set_gravity_on(on)
        ctrl = main_window._active_panel().controls
        if hasattr(ctrl, "chk_gravity"):
            ctrl.chk_gravity.blockSignals(True)
            ctrl.chk_gravity.setChecked(on)
            ctrl.chk_gravity.blockSignals(False)

    ts.gravity_toggled.connect(_fwd_gravity)

    # ── Scale sliders → active panel's pendulum widget ────────────
    def _fwd_overlay(attr: str, value: object) -> None:
        pw = main_window._active_panel().pendulum
        if hasattr(pw, attr):
            getattr(pw, attr)(value)

    ts.force_scale_changed.connect(lambda v: _fwd_overlay("set_force_scale", v))
    ts.mob_scale_changed.connect(lambda v: _fwd_overlay("set_mob_ellipsoid_scale", v))
    ts.force_ell_scale_changed.connect(
        lambda v: _fwd_overlay("set_force_ellipsoid_scale", v)
    )

    # ── Rotation controls (#1146) → active panel's pendulum widget ──
    ts.azimuth_changed.connect(lambda v: _fwd_overlay("set_view_azimuth", v))
    ts.tilt_changed.connect(lambda v: _fwd_overlay("set_tilt_angle", v))

    # ── Reset view → active panel's pendulum widget ───────────────
    ts.reset_view_requested.connect(
        lambda: (
            main_window._active_panel().pendulum.reset_view()
            if hasattr(main_window._active_panel().pendulum, "reset_view")
            else None
        )
    )

    # ── Per-segment overlay visibility ────────────────────────────
    ts.segment_visibility_changed.connect(
        lambda vis: (
            main_window._active_panel().pendulum.set_visible_segments(vis)
            if hasattr(main_window._active_panel().pendulum, "set_visible_segments")
            else None
        )
    )

    # ── Model selection dropdown (#1149) ──────────────────────────
    def _on_model_dropdown_changed(idx: int) -> None:
        main_window._tabs.blockSignals(True)
        main_window._tabs.setCurrentIndex(idx)
        main_window._tabs.blockSignals(False)

    def _on_tab_changed(idx: int) -> None:
        ts.cmb_model.blockSignals(True)
        ts.cmb_model.setCurrentIndex(idx)
        ts.cmb_model.blockSignals(False)

    ts.model_changed.connect(_on_model_dropdown_changed)
    main_window._tabs.currentChanged.connect(_on_tab_changed)

    # ── Busy state and frame sync — only forward from the active panel ─
    # Guard each callback so non-active panels are silently ignored.
    for panel in main_window._panels:
        panel.sim_started.connect(
            lambda _p=panel: (
                ts.set_running(True) if _p is main_window._active_panel() else None
            )
        )
        panel.sim_finished.connect(
            lambda _p=panel: (
                [
                    ts.set_running(False),
                    ts.set_frame_range(_p.current_n_steps()),
                ]
                if _p is main_window._active_panel()
                else None
            )
        )
        panel.frame_changed.connect(
            lambda idx, _p=panel: (
                ts.set_frame(idx) if _p is main_window._active_panel() else None
            )
        )

    # Update segment checkboxes when tab changes
    main_window._tabs.currentChanged.connect(main_window._on_tab_changed)
    # Initialize with the default tab's segments
    main_window._on_tab_changed(main_window._tabs.currentIndex())


def _connect_common_signals(main_window: Any) -> None:
    """Connect common overlay toggle signals to the active panel's pendulum widget.

    This factored-out helper avoids duplication of the same signal wiring
    pattern across the three panel builders.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance containing the toolstrip and panels.
    """
    ts = main_window._toolstrip

    def _fwd_overlay(attr: str, value: object) -> None:
        pw = main_window._active_panel().pendulum
        if hasattr(pw, attr):
            getattr(pw, attr)(value)

    ts.forces_toggled.connect(lambda v: _fwd_overlay("set_show_forces", v))
    ts.zero_torque_toggled.connect(
        lambda v: _fwd_overlay("set_show_zero_torque_forces", v)
    )
    ts.mob_ellipsoid_toggled.connect(
        lambda v: _fwd_overlay("set_show_mob_ellipsoids", v)
    )
    ts.force_ellipsoid_toggled.connect(
        lambda v: _fwd_overlay("set_show_force_ellipsoids", v)
    )
    ts.com_toggled.connect(lambda v: _fwd_overlay("set_show_com", v))

    # ── 3D segment rendering (#1155) ──────────────────────────────
    ts.mode_3d_toggled.connect(lambda v: _fwd_overlay("set_3d_mode", v))
