"""
Main SimulationPanel widget — UI, signal wiring, and animation playback.

Simulation lifecycle helpers (run / done / error / busy indicators /
optimizer apply) live in ``_lifecycle_mixin``.
Export helpers live in ``_export_mixin``.
The background worker and viewer protocol live in ``_worker``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PyQt6.QtCore import QByteArray, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap  # noqa: F401 — used by subclasses
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QProgressBar,
    QSplitter,
    QWidget,
)

from ..side_panel_tabs import SidePanelTabs
from ._export_mixin import _SimulationExportMixin
from ._lifecycle_mixin import _SimulationLifecycleMixin
from ._worker import _SimViewer  # noqa: F401

if TYPE_CHECKING:
    from ..controls_widget import ControlsWidget
    from ..controls_widget_golfer import ControlsWidgetGolfer
    from ..controls_widget_triple import ControlsWidgetTriple

logger = logging.getLogger(__name__)


class SimulationPanel(_SimulationLifecycleMixin, _SimulationExportMixin, QWidget):
    """Reusable panel that hosts controls, pendulum, and matrix widgets."""

    ANIMATION_INTERVAL_MS = 16  # ~60 fps

    # Tab labels — single source of truth so production and tests agree.
    # Each label uses a BMP-range Unicode prefix (codepoint < U+1F300)
    # so it renders correctly in the default font on Linux/WSL where
    # color emoji fonts are not installed.
    TAB_SETUP = "\u2699 Setup"  # U+2699 gear
    TAB_MASS_MATRIX = "\u229e Mass Matrix"  # U+229E squared plus (matrix grid)
    TAB_PLOTS = "\u223f Plots"  # U+223F sine wave (time series)
    TAB_OPTIMIZER = "\u25ce Optimizer"  # U+25CE bullseye (target)
    TAB_NOISE = "\u2744 Noise"  # U+2744 snowflake (random scatter)

    #: Emitted when ODE integration starts (background thread launched)
    sim_started = pyqtSignal()
    #: Emitted when simulation finishes successfully or with an error
    sim_finished = pyqtSignal()
    #: Emitted each time the displayed frame changes (idx: int)
    frame_changed = pyqtSignal(int)
    #: Emitted when animation playback reaches end (for toolstrip play reset)
    playback_ended = pyqtSignal()

    def __init__(
        self,
        controls: ControlsWidget | ControlsWidgetTriple | ControlsWidgetGolfer,
        pendulum: _SimViewer,
        matrix: _SimViewer,
        params_builder: Callable[[dict], object],
        torque_builder: Callable[[dict], Any],
        state_builder: Callable[[dict], np.ndarray],
        run_simulation: Callable,
        torque_history: _SimViewer | None = None,
        limits_builder: Callable[[dict], Any] | None = None,
        clamp_builder: Callable[[dict], Any] | None = None,
        optimizer: QWidget | None = None,
        objective_builder: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        if controls is None:
            raise ValueError("controls must be provided")
        super().__init__(parent)
        self.controls = controls
        self.pendulum = pendulum
        self.matrix = matrix
        self.torque_history = torque_history
        self.optimizer = optimizer
        self.objective_builder = objective_builder
        self.perturbation_panel: QWidget | None = None
        self._params_builder = params_builder
        self._torque_builder = torque_builder
        self._state_builder = state_builder
        self._run_simulation = run_simulation
        self._limits_builder = limits_builder
        self._clamp_builder = clamp_builder

        self._settings_key: str = "splitter_double"

        self._result: Any | None = None
        self._anim_idx = 0
        self._anim_frac: float = 0.0  # fractional frame accumulator (#1097)
        self._playback_speed = 1.0
        self._sim_dt: float = 0.005  # simulation time step (updated on sim completion)
        self._loop_playback: bool = False  # loop animation when it reaches the end

        self._build_ui()
        self._connect_signals()
        self._setup_timer()

    def _build_ui(self) -> None:
        """Build a 2-pane layout: pendulum graphic + side-panel tabs.

        The pendulum widget is *always visible* on the left. Every other
        panel (Setup, Mass Matrix, Plots, Optimizer, Noise) lives in the
        ``SidePanelTabs`` container on the right. Adding new panels is a
        one-liner — see ``set_perturbation_panel``.
        """
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: pendulum widget (always visible)
        splitter.addWidget(cast("QWidget", self.pendulum))

        # RIGHT: tabbed side panels
        self._side_tabs = SidePanelTabs(settings_key=f"{self._settings_key}/active_tab")
        self._side_tabs.add_panel(
            self.TAB_SETUP,
            self.controls,
            tooltip="Configure simulation parameters and run controls",
        )
        self._side_tabs.add_panel(
            self.TAB_MASS_MATRIX,
            cast("QWidget", self.matrix),
            tooltip="Real-time mass matrix and energy display",
        )
        if self.torque_history is not None:
            self._side_tabs.add_panel(
                self.TAB_PLOTS,
                cast("QWidget", self.torque_history),
                tooltip="Torque, energy, and force time-series plots",
            )
        if self.optimizer is not None:
            self._side_tabs.add_panel(
                self.TAB_OPTIMIZER,
                cast("QWidget", self.optimizer),
                tooltip="Gradient-free torque profile optimization",
            )
        splitter.addWidget(self._side_tabs)

        # Pendulum graphic dominates the layout (~60-65 % of width)
        screen = QApplication.primaryScreen()
        sw = screen.availableGeometry().width() if screen else 1400
        side_w = max(360, min(560, int(sw * 0.30)))
        pend_w = max(400, sw - side_w - 20)
        splitter.setSizes([pend_w, side_w])
        splitter.setStretchFactor(0, 4)  # graphic dominant
        splitter.setStretchFactor(1, 2)  # tabs hold their width
        splitter.setCollapsible(0, False)  # never collapse the graphic
        splitter.setCollapsible(1, False)

        main_layout.addWidget(splitter)
        self._splitter = splitter  # keep reference for save/restore

        # Progress bar (hidden by default, positioned absolutely like busy indicator)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #404070; border-radius: 3px; "
            "background: #1a1a2e; padding: 2px; }"
            "QProgressBar::chunk { background: #4a7bdb; }"
        )
        self._progress_bar.setFixedSize(200, 24)
        self._progress_bar.hide()

        # Restore saved splitter state
        settings = QSettings("D-sorganization", "PendulumSimulator")
        saved = settings.value(self._settings_key)
        if isinstance(saved, QByteArray):
            self._splitter.restoreState(saved)
        # Restore the active tab last so the user lands where they left off
        self._side_tabs.restore_state()

    def set_perturbation_panel(self, panel: QWidget) -> None:
        """Attach a perturbation panel as a new tab in the side panel.

        Adds the Noise (Monte Carlo perturbation) tab to the right-hand
        ``SidePanelTabs``. Must be called after construction.

        Pre: ``panel`` is not None.
        Post: ``self.perturbation_panel is panel`` and the Noise tab is
              the last entry in ``self._side_tabs.panel_labels()``.
        """
        if panel is None:
            raise ValueError("perturbation panel must not be None")
        self.perturbation_panel = panel
        self._side_tabs.add_panel(
            self.TAB_NOISE,
            panel,
            tooltip="Monte Carlo noise injection and consistency analysis",
        )

    def save_layout(self) -> None:
        """Persist splitter positions and active side-panel tab to QSettings."""
        settings = QSettings("D-sorganization", "PendulumSimulator")
        settings.setValue(self._settings_key, self._splitter.saveState())
        if hasattr(self, "_side_tabs"):
            self._side_tabs.save_state()

    def _connect_signals(self) -> None:
        self.controls.run_requested.connect(self._on_run)
        self.controls.reset_requested.connect(self._on_reset)
        self.controls.play_toggled.connect(self._on_play_toggle)
        self.controls.speed_changed.connect(self._on_speed_change)
        self.controls.frame_changed.connect(self._on_frame_change)
        self.controls.export_data_requested.connect(self._on_export_data)
        self.controls.export_video_requested.connect(self._on_export_video)
        self.controls.export_image_requested.connect(self.export_image)

        # Wire new physics/display toggles if the pendulum widget supports them
        if hasattr(self.controls, "gravity_changed") and hasattr(
            self.pendulum,
            "set_gravity_on",
        ):
            self.controls.gravity_changed.connect(self.pendulum.set_gravity_on)
        if hasattr(self.controls, "forces_changed") and hasattr(
            self.pendulum,
            "set_show_forces",
        ):
            self.controls.forces_changed.connect(self.pendulum.set_show_forces)
        if hasattr(self.controls, "force_scale_changed") and hasattr(
            self.pendulum,
            "set_force_scale",
        ):
            self.controls.force_scale_changed.connect(self.pendulum.set_force_scale)

        # Wire real-time rotation controls (#1146)
        if hasattr(self.controls, "tilt_changed") and hasattr(
            self.pendulum, "set_tilt_angle"
        ):
            self.controls.tilt_changed.connect(self.pendulum.set_tilt_angle)
        if hasattr(self.controls, "azimuth_changed") and hasattr(
            self.pendulum, "set_view_azimuth"
        ):
            self.controls.azimuth_changed.connect(self.pendulum.set_view_azimuth)

        # Persist splitter when it changes
        if hasattr(self, "_splitter"):
            self._splitter.splitterMoved.connect(lambda *_: self.save_layout())

        # Wire optimizer (#1151): build objective from current params on demand
        if self.optimizer is not None and self.objective_builder is not None:
            from ..optimization_widget import OptimizationWidget

            opt = cast("OptimizationWidget", self.optimizer)
            _obj_builder = self.objective_builder  # capture for closure
            assert callable(_obj_builder), "objective_builder must be callable"

            # Before each optimizer run, rebuild the objective with current params
            orig_on_run = opt._on_run

            def _patched_on_run() -> None:
                """Build objective from current UI params, then run optimizer."""
                try:
                    p = self.controls.get_params()
                    obj_fn = _obj_builder(p)
                    opt.set_objective_function(obj_fn)
                except (ValueError, AssertionError) as e:
                    opt._log.append(f"⚠ Cannot build objective: {e}")
                    return
                orig_on_run()

            opt._btn_run.clicked.disconnect()
            opt._btn_run.clicked.connect(_patched_on_run)

            # Wire apply back to controls (set torque coefficients)
            opt.optimized_coefficients.connect(self._apply_optimized_coefficients)

    def _setup_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.setInterval(self.ANIMATION_INTERVAL_MS)
        self._timer.timeout.connect(self._advance_frame)

    # ------------------------------------------------------------------
    # Playback / animation
    # ------------------------------------------------------------------

    def _on_reset(self) -> None:
        self._timer.stop()
        self._result = None
        self._anim_idx = 0
        self._anim_frac = 0.0
        self.pendulum.clear()
        self.matrix.clear()
        if self.torque_history is not None:
            self.torque_history.clear()
        self.controls.stop_playback()
        self.controls.set_slider_value(0)

    def _on_play_toggle(self, playing: bool) -> None:
        if self._result is None:
            self.controls.stop_playback()
            return
        if playing:
            if self._anim_idx >= self._result.n_steps - 1:
                self._anim_idx = 0
                self._anim_frac = 0.0
                if hasattr(self.pendulum, "_trail"):
                    self.pendulum._trail.clear()
            self._timer.start()
        else:
            self._timer.stop()

    def _on_speed_change(self, speed: float) -> None:
        """Pre: speed > 0"""
        assert speed > 0, f"Playback speed must be positive, got {speed}"
        self._playback_speed = speed

    def _on_frame_change(self, frame: int) -> None:
        if frame is None:
            raise ValueError("frame must be provided")
        if self._result is None:
            return
        self._anim_idx = frame
        if hasattr(self.pendulum, "_trail"):
            self.pendulum._trail.clear()
            trail_len = getattr(self.pendulum, "TRAIL_LENGTH", 200)
            trail_start = max(0, frame - trail_len)
            # Fast path: use vectorized cache if available
            cache = getattr(self.pendulum, "_tip_positions_cache", None)
            if cache is not None:
                for i in range(trail_start, frame + 1):
                    self.pendulum._trail.append(tuple(cache[i]))
            else:
                for i in range(trail_start, frame + 1):
                    pos = self._result.positions_at(i)
                    tip = pos.get("tip")
                    if tip is not None:
                        self.pendulum._trail.append(tip)
        self._display_frame(frame)

    def _advance_frame(self) -> None:
        """Advance the animation by a fractional frame count.

        Uses a fractional accumulator so that high-speed playback (e.g. 5×)
        still produces smooth visual transitions instead of integer jumps.

        Closes #1097.
        """
        if self._result is None:
            self._timer.stop()
            return

        # Inv: _playback_speed > 0, _sim_dt > 0
        assert self._playback_speed > 0, "Playback speed invariant violated"

        # Compute real-time frame advance (#1115)
        # frames_per_tick = wall_clock_tick / sim_dt × speed_multiplier
        dt_wall = self.ANIMATION_INTERVAL_MS / 1000.0
        frames_per_tick = (dt_wall / max(self._sim_dt, 1e-6)) * self._playback_speed
        self._anim_frac += frames_per_tick
        advance = int(self._anim_frac)
        if advance < 1:
            return  # sub-frame accumulation — wait for next tick
        self._anim_frac -= advance

        self._anim_idx += advance

        if self._anim_idx >= self._result.n_steps:
            if self._loop_playback:
                # Loop: restart from beginning
                self._anim_idx = 0
                self._anim_frac = 0.0
            else:
                # Stop at end and reset play button
                self._anim_idx = self._result.n_steps - 1
                self._anim_frac = 0.0
                self._timer.stop()
                self.controls.stop_playback()
                self.playback_ended.emit()

        self._display_frame(self._anim_idx)
        self.controls.set_slider_value(self._anim_idx)

    def _display_frame(self, idx: int) -> None:
        assert self._result is not None
        idx = max(0, min(idx, self._result.n_steps - 1))
        self.pendulum.set_frame(idx)
        self.matrix.set_frame(idx)
        if self.torque_history is not None:
            self.torque_history.set_frame(idx)
        self.frame_changed.emit(idx)

    def scrub_to_frame(self, idx: int) -> None:
        """Jump to a specific frame index (called by toolstrip slider)."""
        if idx is None:
            raise ValueError("idx must be provided")
        if self._result is None:
            return
        idx = max(0, min(idx, self._result.n_steps - 1))
        self._anim_idx = idx
        self.controls.set_slider_value(idx)
        self._display_frame(idx)

    def current_n_steps(self) -> int:
        """Return the number of frames in the current simulation (0 if none)."""
        if self._result is None:
            return 0
        return int(self._result.n_steps)
