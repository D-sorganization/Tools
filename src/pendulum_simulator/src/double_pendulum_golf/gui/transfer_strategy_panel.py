"""Phase-window diagnostics for drift-mediated club transfer.

The widget is intentionally thin: physics reconstruction and integration stay
in :mod:`double_pendulum_golf.transfer_strategy`. Unsupported model tiers fail
closed instead of relabeling hub motion as anatomical shoulder or torso motion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..transfer_strategy import (
    TransferSignals,
    TransferSummary,
    double_pendulum_force_attribution,
    double_pendulum_transfer_signals,
    summarize_transfer,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


_METRICS = (
    ("distal_energy_gain_j", "Distal Energy Gain", "J"),
    ("drift_grip_work_j", "Drift Grip Work", "J"),
    ("control_grip_work_j", "Control Grip Work", "J"),
    ("negative_grip_work_j", "Negative Grip Work", "J"),
    ("wrist_control_work_j", "Wrist Control Work", "J"),
    ("peak_grip_force_n", "Peak Grip Force", "N"),
    ("peak_distal_speed_m_s", "Peak Distal Speed", "m/s"),
)

_SOURCE_COMPONENTS = (
    ("coriolis", "Coriolis Cross-Speed"),
    ("squared_speed", "Squared-Speed / Centripetal"),
    ("gravity", "Gravity"),
    ("damping", "Damping"),
    ("applied", "Applied Drive"),
)


class TransferStrategyPanel:
    """Display transfer metrics over one user-declared trajectory window."""

    def __init__(self, parent: Any = None) -> None:
        from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

        self._signals: TransferSignals | None = None
        self._attribution: Any = None
        self._source_warning: str | None = None
        self._last_summary: TransferSummary | None = None
        self._widget = QWidget(parent)
        layout = QVBoxLayout(self._widget)

        self._status = QLabel("Run a simulation to evaluate drift-mediated transfer.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addWidget(self._build_window_group())
        layout.addWidget(self._build_metric_group())
        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_plot_widget())

        claim = QLabel(
            "Model boundary: θ̇₁ is proximal-link angular velocity. It is not an "
            "anatomical shoulder-joint or torso measurement."
        )
        claim.setWordWrap(True)
        layout.addWidget(claim)

    def _build_window_group(self) -> Any:
        from PyQt6.QtWidgets import (
            QDoubleSpinBox,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
        )

        group = QGroupBox("Declared Analysis Window")
        layout = QHBoxLayout(group)
        self._start_spin = QDoubleSpinBox()
        self._end_spin = QDoubleSpinBox()
        for spin in (self._start_spin, self._end_spin):
            spin.setDecimals(4)
            spin.setSuffix(" s")
            spin.setKeyboardTracking(False)
        layout.addWidget(QLabel("Start:"))
        layout.addWidget(self._start_spin)
        layout.addWidget(QLabel("End:"))
        layout.addWidget(self._end_spin)
        button = QPushButton("Evaluate Transfer")
        button.clicked.connect(self.refresh)
        layout.addWidget(button)
        return group

    def _build_metric_group(self) -> Any:
        from PyQt6.QtWidgets import QGridLayout, QGroupBox, QLabel

        group = QGroupBox("Transfer Outcomes")
        layout = QGridLayout(group)
        self._metric_labels: dict[str, QLabel] = {}
        for index, (key, description, unit) in enumerate(_METRICS):
            name = QLabel(f"{description}:")
            value = QLabel(f"-- {unit}")
            layout.addWidget(name, index // 2, 2 * (index % 2))
            layout.addWidget(value, index // 2, 2 * (index % 2) + 1)
            self._metric_labels[key] = value
        return group

    def _build_source_group(self) -> Any:
        from PyQt6.QtWidgets import QGridLayout, QGroupBox, QLabel

        group = QGroupBox("Full-Trajectory Coordinate Force Sources")
        layout = QGridLayout(group)
        for column, heading in enumerate(
            ("Source", "Signed tangent impulse", "Absolute tangent impulse", "Work")
        ):
            layout.addWidget(QLabel(heading), 0, column)
        self._source_labels: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        for row, (key, description) in enumerate(_SOURCE_COMPONENTS, start=1):
            layout.addWidget(QLabel(description), row, 0)
            values = (QLabel("-- N s"), QLabel("-- N s"), QLabel("-- J"))
            for column, value in enumerate(values, start=1):
                layout.addWidget(value, row, column)
            self._source_labels[key] = values
        note = QLabel(
            "Christoffel cross/squared split uses shoulder-absolute and "
            "wrist-relative coordinates. Values are modeled equation-term "
            "attributions, not measured muscle forces."
        )
        note.setWordWrap(True)
        layout.addWidget(note, len(_SOURCE_COMPONENTS) + 1, 0, 1, 4)
        return group

    def _build_plot_widget(self) -> Any:
        from PyQt6.QtWidgets import QLabel

        if not _HAS_MPL:
            self._figure = None
            return QLabel("Install matplotlib to view transfer traces.")
        self._figure = Figure(figsize=(7, 4), dpi=100)
        self._axis_power = self._figure.add_subplot(211)
        self._axis_speed = self._figure.add_subplot(212, sharex=self._axis_power)
        self._canvas = FigureCanvasQTAgg(self._figure)
        return self._canvas

    def widget(self) -> Any:
        """Return the embeddable Qt widget."""
        return self._widget

    def set_result(self, result: Any, model_type: str) -> None:
        """Load one qualified result or fail closed for unsupported tiers."""
        self._last_summary = None
        self._attribution = None
        self._source_warning = None
        if model_type != "double":
            self._signals = None
            self._status.setText(
                f"Transfer attribution is not yet qualified for the {model_type} model."
            )
            self._clear_metrics()
            self._clear_sources()
            return
        try:
            self._signals = double_pendulum_transfer_signals(result)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._signals = None
            self._status.setText(f"Transfer diagnostics unavailable: {exc}")
            self._clear_metrics()
            self._clear_sources()
            return
        try:
            self._attribution = double_pendulum_force_attribution(result)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._attribution = None
            self._clear_sources()
            self._source_warning = str(exc)
        else:
            self._render_sources()
        start = float(self._signals.time_s[0])
        end = float(self._signals.time_s[-1])
        for spin in (self._start_spin, self._end_spin):
            spin.setRange(start, end)
            spin.setSingleStep(max((end - start) / 100.0, 1e-4))
        self._start_spin.setValue(start)
        self._end_spin.setValue(end)
        self.refresh()

    def refresh(self) -> None:
        """Recompute and redraw the selected phase window."""
        if self._signals is None:
            return
        start = self._start_spin.value()
        end = self._end_spin.value()
        if start >= end:
            self._last_summary = None
            self._status.setText("Window start must precede end.")
            self._clear_metrics()
            return
        summary = summarize_transfer(self._signals, start_s=start, end_s=end)
        self._last_summary = summary
        if self._source_warning is None:
            self._status.setText("Exact Planar Double Pendulum")
        else:
            self._status.setText(
                "Exact Planar Double Pendulum; coordinate sources unavailable: "
                f"{self._source_warning}"
            )
        for key, _, unit in _METRICS:
            value = getattr(summary, key)
            self._metric_labels[key].setText(f"{value:.3f} {unit}")
        self._draw_window(start, end)

    def _clear_metrics(self) -> None:
        for key, _, unit in _METRICS:
            self._metric_labels[key].setText(f"-- {unit}")

    def _clear_sources(self) -> None:
        for values in self._source_labels.values():
            values[0].setText("-- N s")
            values[1].setText("-- N s")
            values[2].setText("-- J")

    def _render_sources(self) -> None:
        if self._attribution is None:
            return
        for key, _ in _SOURCE_COMPONENTS:
            metric = self._attribution.metrics[key]
            signed = metric.signed_tangent_impulse_n_s
            absolute = metric.absolute_tangent_impulse_n_s
            values = self._source_labels[key]
            values[0].setText("undefined" if signed is None else f"{signed:.3f} N s")
            values[1].setText("undefined" if absolute is None else f"{absolute:.3f} N s")
            values[2].setText(f"{metric.generalized_work_j:.3f} J")

    def _draw_window(self, start: float, end: float) -> None:
        if self._figure is None or self._signals is None:
            return
        signals = self._signals
        mask = (signals.time_s >= start) & (signals.time_s <= end)
        time = signals.time_s[mask]
        velocity = signals.grip_velocity_m_s[mask]
        drift_power = np.einsum("ij,ij->i", signals.grip_force_drift_n[mask], velocity)
        control_power = np.einsum("ij,ij->i", signals.grip_force_control_n[mask], velocity)
        total_power = drift_power + control_power

        power_axis = self._axis_power
        speed_axis = self._axis_speed
        power_axis.clear()
        speed_axis.clear()
        power_axis.plot(time, drift_power, label="Drift Grip Power")
        power_axis.plot(time, control_power, label="Control Grip Power")
        power_axis.plot(time, total_power, label="Total Grip Power", linewidth=2.0)
        power_axis.fill_between(time, total_power, 0.0, where=total_power < 0.0, alpha=0.2)
        power_axis.axhline(0.0, color="black", linewidth=0.7)
        power_axis.set_ylabel("Power (W)")
        power_axis.legend(loc="best", fontsize=8)

        speed_axis.plot(time, signals.distal_speed_m_s[mask], label="Distal Speed")
        speed_axis.plot(
            time,
            signals.proximal_angular_velocity_rad_s[mask],
            label="Proximal Angular Velocity",
        )
        speed_axis.set_xlabel("Time (s)")
        speed_axis.set_ylabel("Speed / Rate")
        speed_axis.legend(loc="best", fontsize=8)
        self._figure.tight_layout()
        self._canvas.draw_idle()


__all__ = ["TransferStrategyPanel"]
