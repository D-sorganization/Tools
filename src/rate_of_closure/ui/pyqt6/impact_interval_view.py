"""Sub-microsecond impact-history plots and slow-motion contact view."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from rate_of_closure.simulation import SimulationRun
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M

try:
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover

    def get_chart_color(index: int) -> str:
        return f"C{index % 10}"


__all__ = ["ImpactIntervalView"]

_SLIDER_STEPS = 1_000


class ImpactIntervalView(QWidget):
    """Queryable impact interval with force, orientation, and state views."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._run: SimulationRun | None = None
        self._figure = Figure(figsize=(8, 6), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.subplots(2, 2)
        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setRange(0, _SLIDER_STEPS)
        self._position_slider.setEnabled(False)
        self._position_slider.setToolTip(
            "Scrub the recorded sub-microsecond club/ball state through contact. "
            "Source: the selected Impact Interval solver history."
        )
        self._position_slider.valueChanged.connect(self._draw)
        self._time_label = QLabel("No impact-interval history")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Contact Playback"))
        controls.addWidget(self._position_slider, stretch=1)
        controls.addWidget(self._time_label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self._canvas)
        self._draw()

    def set_run(self, run: SimulationRun | None) -> None:
        """Adopt a simulation run and expose its interval history, if present."""
        self._run = run
        enabled = run is not None and run.impact_interval is not None
        self._position_slider.setEnabled(enabled)
        self._position_slider.blockSignals(True)
        self._position_slider.setValue(0)
        self._position_slider.blockSignals(False)
        self._draw()

    def run(self) -> SimulationRun | None:
        """The run currently displayed."""
        return self._run

    def _sample_index(self) -> int:
        interval = self._run.impact_interval if self._run is not None else None
        if interval is None:
            return 0
        fraction = self._position_slider.value() / _SLIDER_STEPS
        return min(
            round(fraction * (len(interval.time_s) - 1)), len(interval.time_s) - 1
        )

    def _draw_scene(self, index: int) -> None:
        assert self._run is not None and self._run.impact_interval is not None
        interval = self._run.impact_interval
        axes = self._axes[1, 0]
        origin = interval.ball_position_m[0]
        ball = (interval.ball_position_m[index] - origin) * 1_000.0
        contact = (interval.contact_point_position_m[index] - origin) * 1_000.0
        normal = interval.contact_normal[index]
        axes.add_patch(
            Circle(
                (ball[0], ball[1]),
                GOLF_BALL_RADIUS_M * 1_000.0,
                color=get_chart_color(4),
                alpha=0.35,
            )
        )
        tangent = np.array([-normal[1], normal[0]])
        face = np.vstack([contact[:2] - 30.0 * tangent, contact[:2] + 30.0 * tangent])
        axes.plot(face[:, 0], face[:, 1], color=get_chart_color(1), lw=3.0)
        axes.scatter(contact[0], contact[1], color=get_chart_color(3), s=30)
        axes.arrow(
            contact[0],
            contact[1],
            12.0 * normal[0],
            12.0 * normal[1],
            color=get_chart_color(0),
            width=0.4,
            length_includes_head=True,
        )
        axes.set_aspect("equal")
        axes.set_xlim(-45.0, 45.0)
        axes.set_ylim(-45.0, 45.0)
        axes.set_xlabel("Target Direction [mm]")
        axes.set_ylabel("Up [mm]")
        axes.set_title("Slow-Motion Contact State")

    def _draw_audit(self) -> None:
        assert self._run is not None and self._run.impact_interval is not None
        audit = self._run.impact_interval.audit
        axes = self._axes[1, 1]
        axes.axis("off")
        lines = (
            "Scientific Audit",
            f"Normal Impulse: {audit.integrated_normal_impulse_n_s:.5f} N·s",
            f"Friction Impulse: {audit.integrated_friction_impulse_n_s:.5f} N·s",
            f"Dissipated Energy: {audit.dissipated_energy_j:.4f} J",
            f"Energy Residual: {audit.energy_residual_j:+.3e} J",
            f"Momentum Residual: {audit.linear_momentum_residual_n_s:.3e} N·s",
        )
        axes.text(0.03, 0.95, "\n".join(lines), va="top", family="monospace")

    def _draw(self, _value: int = 0) -> None:
        for axes in self._axes.flat:
            axes.clear()
        interval = self._run.impact_interval if self._run is not None else None
        if interval is None:
            self._axes[0, 0].text(
                0.5,
                0.5,
                "Select Impact Interval (6-DOF) and run the simulation.",
                ha="center",
                va="center",
            )
            self._axes[0, 0].set_axis_off()
            self._time_label.setText("No impact-interval history")
            self._canvas.draw()
            return

        index = self._sample_index()
        time_us = interval.time_s * 1.0e6
        now_us = float(time_us[index])
        force_axes = self._axes[0, 0]
        force_axes.plot(
            time_us, interval.normal_force_n / 1_000.0, color=get_chart_color(0)
        )
        force_axes.axvline(now_us, color=get_chart_color(7), ls="--")
        force_axes.set_xlabel("Time [µs]")
        force_axes.set_ylabel("Normal Force [kN]")
        force_axes.set_title("Contact Force History")

        angle_axes = self._axes[0, 1]
        angle_axes.plot(time_us, interval.face_angle_deg, label="Face Angle")
        angle_axes.plot(time_us, interval.dynamic_loft_deg, label="Dynamic Loft")
        angle_axes.plot(
            time_us,
            np.degrees(interval.twist_angle_rad),
            label="Shaft-Axis Twist",
        )
        angle_axes.axvline(now_us, color=get_chart_color(7), ls="--")
        angle_axes.set_xlabel("Time [µs]")
        angle_axes.set_ylabel("Angle [deg]")
        angle_axes.set_title("Loaded Face Motion")
        angle_axes.legend(fontsize=7)

        self._draw_scene(index)
        self._draw_audit()
        self._time_label.setText(
            f"{now_us:.1f} µs · {interval.normal_force_n[index] / 1_000.0:.2f} kN"
        )
        self._canvas.draw()
