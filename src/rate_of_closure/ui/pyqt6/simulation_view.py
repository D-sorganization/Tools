"""3D scene + full video playback controls for the simulation session.

Renders one :class:`~rate_of_closure.simulation.session.SimulationRun`:
the swing path with the clubhead marker at the playback instant, the
fixed ball (own checkbox), the ground plane (own checkbox), the flight
trajectory polyline, and a toggleable instantaneous-screw-axis overlay
computed through the one thin ISA adapter (recon #4108).

Playback is a full video bar — play/pause, a scrub slider over the
whole swing + flight timeline, frame step +/-, a loop toggle, and rate
presets where 1x maps animation wall time to simulated time.

Colors come from the shared UpstreamDrift theme palette
(``get_chart_color``); no app colors are hard-coded here.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.simulation import (
    BALL_POSITION_M,
    SimulationRun,
    screw_axis_samples,
)
from rate_of_closure.simulation.isa import MIN_RATE_DPS
from rate_of_closure.units import FIELD_GUIDANCE

try:  # Theme palette (optional in standalone/vendored use).
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover - theme package always ships in-repo

    def get_chart_color(index: int) -> str:
        """Matplotlib cycle colors as a theme-neutral fallback."""
        return f"C{index % 10}"


logger = logging.getLogger(__name__)

__all__ = ["RATE_PRESETS", "SimulationView"]

#: Playback-rate presets, in combo order. 1x is real time: one second
#: of wall clock advances one second of simulated time.
RATE_PRESETS: tuple[tuple[str, float], ...] = (
    ("0.1×", 0.1),
    ("0.25×", 0.25),
    ("0.5×", 0.5),
    ("1× real-time", 1.0),
    ("2×", 2.0),
)

_TIMER_INTERVAL_MS = 40
_SLIDER_STEPS = 1000
_BALL_DRAW_RADIUS_M = 0.03  # slightly over scale so the ball stays visible


class SimulationView(QWidget):
    """Animated 3D scene of one simulation run with video controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")

        self._run: SimulationRun | None = None
        self._screws: list[dict] | None = None
        self._time = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._build_playback_bar())
        layout.addLayout(self._build_toggle_bar())
        layout.addWidget(self._canvas)

        self._timer = QTimer(self)
        self._timer.setInterval(_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)

    # ── construction ────────────────────────────────────────────────
    def _build_playback_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 4, 4, 0)

        self._play_button = QPushButton("Play")
        self._play_button.setCheckable(True)
        self._play_button.setFixedWidth(64)
        self._play_button.setToolTip("Play or pause the swing + flight playback.")
        self._play_button.toggled.connect(self._on_play_toggled)
        bar.addWidget(self._play_button)

        self._step_back_button = QPushButton("−1 frame")
        self._step_back_button.setToolTip("Step one sample backward.")
        self._step_back_button.clicked.connect(lambda: self.step_frames(-1))
        bar.addWidget(self._step_back_button)

        self._step_forward_button = QPushButton("+1 frame")
        self._step_forward_button.setToolTip("Step one sample forward.")
        self._step_forward_button.clicked.connect(lambda: self.step_frames(1))
        bar.addWidget(self._step_forward_button)

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setRange(0, _SLIDER_STEPS)
        self._position_slider.setToolTip(
            "Scrub the playback instant across the whole swing + flight timeline."
        )
        self._position_slider.valueChanged.connect(self._on_slider_moved)
        bar.addWidget(self._position_slider, stretch=1)

        self._time_label = QLabel("0.000 s")
        self._time_label.setFixedWidth(72)
        bar.addWidget(self._time_label)

        self._loop_check = QCheckBox("Loop")
        self._loop_check.setToolTip("Restart playback when the timeline ends.")
        bar.addWidget(self._loop_check)

        bar.addWidget(QLabel("Rate"))
        self._rate_combo = QComboBox()
        self._rate_combo.addItems([name for name, _ in RATE_PRESETS])
        self._rate_combo.setCurrentIndex(3)  # 1x real-time
        self._rate_combo.setToolTip(
            "Playback rate: 1× maps animation time to simulated time; "
            "slower presets reveal the impact window."
        )
        bar.addWidget(self._rate_combo)
        return bar

    def _build_toggle_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 0)
        self._ball_check = QCheckBox("Ball")
        self._ball_check.setChecked(True)
        self._ball_check.setToolTip(FIELD_GUIDANCE["ball_visible"])
        self._ground_check = QCheckBox("Ground")
        self._ground_check.setChecked(True)
        self._ground_check.setToolTip(FIELD_GUIDANCE["ground_visible"])
        self._screw_check = QCheckBox("Screw Axis")
        self._screw_check.setChecked(False)
        self._screw_check.setToolTip(FIELD_GUIDANCE["screw_axis_visible"])
        for check in (self._ball_check, self._ground_check, self._screw_check):
            check.toggled.connect(lambda _checked: self._draw())
            bar.addWidget(check)
        bar.addStretch(1)
        return bar

    # ── public API ──────────────────────────────────────────────────
    def set_run(self, run: SimulationRun | None) -> None:
        """Adopt a run (or clear with ``None``) and reset the timeline."""
        self._run = run
        self._screws = None
        self._time = 0.0
        self._sync_slider()
        self._draw()

    def run(self) -> SimulationRun | None:
        """The run currently rendered, if any."""
        return self._run

    def playback_time(self) -> float:
        """Current playback instant [s] on the swing + flight timeline."""
        return self._time

    def set_playback_time(self, t: float) -> None:
        """Move the playback instant (clamped to the timeline)."""
        if self._run is None:
            return
        self._time = min(max(t, 0.0), self._run.total_duration_s)
        self._sync_slider()
        self._draw()

    def playback_rate(self) -> float:
        """The selected playback-rate multiplier."""
        return RATE_PRESETS[self._rate_combo.currentIndex()][1]

    def set_playback_rate(self, multiplier: float) -> None:
        """Select the nearest rate preset to ``multiplier``."""
        index = int(np.argmin([abs(rate - multiplier) for _, rate in RATE_PRESETS]))
        self._rate_combo.setCurrentIndex(index)

    def step_frames(self, frames: int) -> None:
        """Step the playback instant by whole swing-sample intervals."""
        if self._run is None:
            return
        dt = float(self._run.swing_times[1] - self._run.swing_times[0])
        self.set_playback_time(self._time + frames * dt)

    def is_playing(self) -> bool:
        """Whether the playback timer is running."""
        return self._timer.isActive()

    def set_looping(self, looping: bool) -> None:
        """Set the loop toggle."""
        self._loop_check.setChecked(looping)

    def stop(self) -> None:
        """Stop the playback timer (window close and tests)."""
        self._timer.stop()
        self._play_button.setChecked(False)

    # ── internals ──────────────────────────────────────────────────
    def _on_play_toggled(self, playing: bool) -> None:
        self._play_button.setText("Pause" if playing else "Play")
        if playing and self._run is not None:
            self._timer.start()
        else:
            self._timer.stop()

    def _on_slider_moved(self, value: int) -> None:
        if self._run is None:
            return
        t = value / _SLIDER_STEPS * self._run.total_duration_s
        if abs(t - self._time) > 1e-12:
            self._time = t
            self._draw()

    def _sync_slider(self) -> None:
        total = self._run.total_duration_s if self._run is not None else 0.0
        value = round(self._time / total * _SLIDER_STEPS) if total > 0.0 else 0
        self._position_slider.blockSignals(True)
        self._position_slider.setValue(value)
        self._position_slider.blockSignals(False)
        self._time_label.setText(f"{self._time:.3f} s")

    def _advance(self) -> None:
        if self._run is None:
            return
        self._time += _TIMER_INTERVAL_MS / 1000.0 * self.playback_rate()
        total = self._run.total_duration_s
        if self._time > total:
            if self._loop_check.isChecked():
                self._time = 0.0
            else:
                self._time = total
                self._play_button.setChecked(False)
        self._sync_slider()
        self._draw()

    def _screw_entries(self) -> list[dict]:
        if self._run is None:
            return []
        if self._screws is None:
            dt = float(self._run.swing_times[1] - self._run.swing_times[0])
            self._screws = screw_axis_samples(self._run.swing_poses, dt)
        return self._screws

    @staticmethod
    def _display(points: np.ndarray) -> np.ndarray:
        """App frame (x target, y up, z right) -> matplotlib display axes."""
        return np.asarray(points)[..., [2, 0, 1]]

    def _draw_ball(self) -> None:
        u = np.linspace(0.0, 2.0 * np.pi, 16)
        v = np.linspace(0.0, np.pi, 12)
        r = _BALL_DRAW_RADIUS_M
        x = BALL_POSITION_M[0] + r * np.outer(np.cos(u), np.sin(v))
        y = BALL_POSITION_M[1] + r * np.outer(np.sin(u), np.sin(v))
        z = BALL_POSITION_M[2] + r * np.outer(np.ones_like(u), np.cos(v))
        pts = self._display(np.stack([x, y, z], axis=-1))
        self._axes.plot_surface(
            pts[..., 0],
            pts[..., 1],
            pts[..., 2],
            color=get_chart_color(4),
            alpha=0.9,
            linewidth=0.0,
            shade=True,
        )

    def _draw_ground(self, extent: float) -> None:
        grid = np.linspace(-extent, extent, 2)
        gx, gz = np.meshgrid(grid, grid)
        gy = np.zeros_like(gx)
        pts = self._display(np.stack([gx, gy, gz], axis=-1))
        self._axes.plot_surface(
            pts[..., 0],
            pts[..., 1],
            pts[..., 2],
            color=get_chart_color(7),
            alpha=0.12,
            linewidth=0.0,
            shade=False,
        )

    def _draw_screw_axis(self, index: int, extent: float) -> None:
        entries = self._screw_entries()
        if not entries:
            return
        entry = entries[min(index, len(entries) - 1)]
        if entry["rate_dps"] < MIN_RATE_DPS:
            return
        axis, point = entry["axis"], entry["point"]
        length = extent * 1.2
        line = np.vstack([point - length * axis, point + length * axis])
        pts = self._display(line)
        self._axes.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=get_chart_color(5),
            lw=1.6,
            ls="--",
            label=(
                f"screw axis — {entry['rate_dps']:.0f} °/s, "
                f"pitch {entry['pitch']:.3f} m/rad, "
                f"R_ISA {entry['r_isa_m']:.2f} m"
            ),
        )

    def _draw(self) -> None:
        axes = self._axes
        elev, azim = float(axes.elev), float(axes.azim)
        axes.clear()
        run = self._run
        if run is None:
            axes.set_title("Run a simulation to populate the scene")
            self._canvas.draw_idle()
            return

        swing_end = float(run.swing_times[-1])
        in_flight = self._time > run.impact_time_s
        index = int(
            np.searchsorted(run.swing_times, min(self._time, swing_end), side="left")
        )
        index = min(index, len(run.swing_times) - 1)

        # Scene extent: the swing envelope, or the flight envelope once
        # the playback instant is past impact.
        if in_flight and len(run.flight_positions):
            extent = max(5.0, float(np.max(np.abs(run.flight_positions))) * 1.05)
        else:
            extent = max(
                1.0,
                float(np.max(np.abs(run.swing_positions))) * 1.1,
            )

        if self._ground_check.isChecked():
            self._draw_ground(extent)
        if self._ball_check.isChecked():
            self._draw_ball()

        # Swing path: full arc faint, traversed portion solid, head marker.
        full = self._display(run.swing_positions)
        axes.plot(
            full[:, 0],
            full[:, 1],
            full[:, 2],
            color=get_chart_color(0),
            lw=0.8,
            alpha=0.35,
        )
        done = self._display(run.swing_positions[: index + 1])
        axes.plot(
            done[:, 0],
            done[:, 1],
            done[:, 2],
            color=get_chart_color(0),
            lw=2.0,
            label="clubhead path",
        )
        head = self._display(run.swing_positions[index])
        axes.scatter(*head, color=get_chart_color(1), s=45, zorder=5)

        # Flight trajectory: traversed portion once past impact.
        if len(run.flight_positions):
            flight_t = self._time - run.impact_time_s
            n_flight = int(np.searchsorted(run.flight_times, flight_t, side="right"))
            traj = self._display(run.flight_positions)
            axes.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                color=get_chart_color(2),
                lw=0.8,
                alpha=0.35,
            )
            if n_flight > 1:
                done_traj = self._display(run.flight_positions[:n_flight])
                axes.plot(
                    done_traj[:, 0],
                    done_traj[:, 1],
                    done_traj[:, 2],
                    color=get_chart_color(2),
                    lw=2.0,
                    label="ball flight",
                )
                ball = self._display(run.flight_positions[n_flight - 1])
                axes.scatter(*ball, color=get_chart_color(4), s=25, zorder=5)

        if self._screw_check.isChecked() and not in_flight:
            self._draw_screw_axis(index, extent)

        axes.set_xlim(-extent, extent)
        axes.set_ylim(-extent, extent)
        axes.set_zlim(0.0 if in_flight else -extent * 0.4, extent)
        axes.view_init(elev=elev, azim=azim)
        axes.set_xlabel("z — right of target [m]")
        axes.set_ylabel("x — target line [m]")
        axes.set_zlabel("y — up [m]")
        phase = "flight" if in_flight else "swing"
        axes.set_title(
            f"t = {self._time:.3f} s ({phase}) — impact at {run.impact_time_s:.3f} s"
        )
        axes.legend(loc="upper left", fontsize=8)
        self._canvas.draw_idle()
