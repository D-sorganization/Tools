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
    QScrollArea,
    QSplitter,
    QWidget,
)

from ._export_mixin import _SimulationExportMixin
from ._lifecycle_mixin import _SimulationLifecycleMixin
from ._worker import _SCROLL_STYLE, _SimViewer

if TYPE_CHECKING:
    from ..controls_widget import ControlsWidget
    from ..controls_widget_golfer import ControlsWidgetGolfer
    from ..controls_widget_triple import ControlsWidgetTriple

logger = logging.getLogger(__name__)


class SimulationPanel(_SimulationLifecycleMixin, _SimulationExportMixin, QWidget):
    """Reusable panel that hosts controls, pendulum, and matrix widgets."""

    ANIMATION_INTERVAL_MS = 16  # ~60 fps

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
        assert controls is not None, "controls must be provided"
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
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Wrap controls in a scroll area so it never clips on small heights
        scroll = QScrollArea()
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(380)
        scroll.setStyleSheet(_SCROLL_STYLE)
        splitter.addWidget(scroll)

        splitter.addWidget(cast("QWidget", self.pendulum))
        splitter.addWidget(cast("QWidget", self.matrix))

        # Use proportional sizes: compute from available screen width
        screen = QApplication.primaryScreen()
        sw = screen.availableGeometry().width() if screen else 1400
        ctrl_w = min(320, int(sw * 0.20))
        matrix_w = min(280, int(sw * 0.18))

        if self.torque_history is not None:
            splitter.addWidget(cast("QWidget", self.torque_history))
            torque_w = min(260, int(sw * 0.16))
            pend_w = sw - ctrl_w - matrix_w - torque_w - 20
            splitter.setSizes([ctrl_w, pend_w, matrix_w, torque_w])
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 3)
            splitter.setStretchFactor(2, 1)
            splitter.setStretchFactor(3, 1)
        else:
            pend_w = sw - ctrl_w - matrix_w - 20
            splitter.setSizes([ctrl_w, pend_w, matrix_w])
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 3)
            splitter.setStretchFactor(2, 1)

        # Add optimizer panel if provided (#1108, #1109, #1110)
        if self.optimizer is not None:
            opt_scroll = QScrollArea()
            opt_scroll.setWidget(self.optimizer)
            opt_scroll.setWidgetResizable(True)
            opt_scroll.setMinimumWidth(200)
            opt_scroll.setMaximumWidth(300)
            opt_scroll.setStyleSheet(_SCROLL_STYLE)
            splitter.addWidget(opt_scroll)
            splitter.setStretchFactor(splitter.count() - 1, 0)

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

    def set_perturbation_panel(self, panel: QWidget) -> None:
        """Attach a perturbation panel to the right side of the splitter.

        Must be called after construction but before the widget is shown.
        """
        assert panel is not None, "perturbation panel must not be None"
        self.perturbation_panel = panel
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(200)
        scroll.setMaximumWidth(320)
        scroll.setStyleSheet(_SCROLL_STYLE)
        self._splitter.addWidget(scroll)
        self._splitter.setStretchFactor(self._splitter.count() - 1, 0)

    def save_layout(self) -> None:
        """Persist the current splitter positions to QSettings."""
        settings = QSettings("D-sorganization", "PendulumSimulator")
        settings.setValue(self._settings_key, self._splitter.saveState())

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
        if hasattr(self.controls, "tilt_changed") and hasattr(self.pendulum, "set_tilt_angle"):
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
        assert frame is not None, "frame must be provided"
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
        assert idx is not None, "idx must be provided"
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
