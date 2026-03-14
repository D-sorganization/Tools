"""Metrics panel — live physics metrics and educational explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from asteroid_jumper.controller import SimController


class MetricsPanel(QWidget):
    """Right-side panel showing live physics metrics and insight bars."""

    def __init__(
        self, controller: SimController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        assert controller is not None
        self._ctrl = controller
        self._max_speed: float = 0.1  # for bar scaling
        self._build_ui()
        self.setMinimumWidth(220)

    def refresh(self) -> None:
        """Update all displayed values from controller."""
        jspeed = self._ctrl.jumper_speed()
        jaw = self._ctrl.jumper_angular_speed()
        aspeed = self._ctrl.asteroid_speed()
        aaw = self._ctrl.asteroid_angular_speed()
        offctr = self._ctrl.off_centre_fraction()
        total_p = self._ctrl.state.total_linear_momentum.length()

        # Update peak scale
        self._max_speed = max(self._max_speed, jspeed, aspeed, 0.1)

        self._jspeed_val.setText(f"{jspeed:.4f} m/s")
        self._jaw_val.setText(f"{jaw:.4f} rad/s")
        self._aspeed_val.setText(f"{aspeed:.4f} m/s")
        self._aaw_val.setText(f"{aaw:.4f} rad/s")
        self._offctr_val.setText(f"{offctr:.1%}")
        self._momentum_val.setText(f"{total_p:.4f} kg·m/s")

        # Bars (0–100%)
        self._jspeed_bar.setValue(int(100 * jspeed / self._max_speed))
        self._jaw_bar.setValue(int(min(100, jaw * 20)))  # rad/s → %
        self._aspeed_bar.setValue(int(100 * aspeed / self._max_speed))
        self._aaw_bar.setValue(int(min(100, aaw * 20)))

        # Efficiency insight
        eff = 1.0 - offctr
        self._eff_bar.setValue(int(100 * eff))
        self._eff_label.setText(
            "Max translational efficiency"
            if eff > 0.95
            else "Spin induced — efficiency reduced"
        )

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(self._build_jumper_group())
        layout.addWidget(self._build_asteroid_group())
        layout.addWidget(self._build_efficiency_group())
        layout.addWidget(self._build_explainer())
        layout.addStretch()

    def _build_jumper_group(self) -> QGroupBox:
        group = QGroupBox("Jumper (global frame)")
        grid = QVBoxLayout(group)

        self._jspeed_val, self._jspeed_bar = _metric_row(
            grid, "Translational v:", "#a6e3a1"
        )
        self._jaw_val, self._jaw_bar = _metric_row(grid, "Angular ω:", "#89b4fa")
        return group

    def _build_asteroid_group(self) -> QGroupBox:
        group = QGroupBox("Asteroid (global frame)")
        grid = QVBoxLayout(group)

        self._aspeed_val, self._aspeed_bar = _metric_row(
            grid, "Translational v:", "#94e2d5"
        )
        self._aaw_val, self._aaw_bar = _metric_row(grid, "Angular ω:", "#cba6f7")
        return group

    def _build_efficiency_group(self) -> QGroupBox:
        group = QGroupBox("Impulse Efficiency")
        vbox = QVBoxLayout(group)

        vbox.addWidget(QLabel("Off-centre ratio:"))
        self._offctr_val = QLabel("0%")
        vbox.addWidget(self._offctr_val)

        vbox.addWidget(QLabel("Translational efficiency:"))
        self._eff_bar = QProgressBar()
        self._eff_bar.setRange(0, 100)
        self._eff_bar.setValue(100)
        self._eff_bar.setStyleSheet(
            "QProgressBar::chunk { background-color: #a6e3a1; }"
        )
        vbox.addWidget(self._eff_bar)
        self._eff_label = QLabel("Max translational efficiency")
        self._eff_label.setWordWrap(True)
        vbox.addWidget(self._eff_label)

        vbox.addWidget(QLabel("Total linear momentum:"))
        self._momentum_val = QLabel("0.0000 kg·m/s")
        vbox.addWidget(self._momentum_val)
        return group

    def _build_explainer(self) -> QGroupBox:
        group = QGroupBox("Physics Insight")
        vbox = QVBoxLayout(group)
        text = QLabel(
            "When the jump force passes through <b>both COMs</b>, all impulse "
            "goes to translation (max Δv, zero spin).\n\n"
            "An off-centre force splits impulse: some goes to rotation (spin) "
            "of both bodies, reducing translational Δv.\n\n"
            "Conservation: total linear momentum stays zero (started at rest)."
        )
        text.setWordWrap(True)
        text.setStyleSheet("font-size: 9pt;")
        vbox.addWidget(text)
        return group


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _metric_row(
    layout: QVBoxLayout,
    label: str,
    bar_color: str,
) -> tuple[QLabel, QProgressBar]:
    """Add a label+value+bar row to *layout*. Returns (value_label, bar)."""
    assert layout is not None, "layout must be provided"
    layout.addWidget(QLabel(label))
    val_label = QLabel("0.0000")
    layout.addWidget(val_label)
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setFixedHeight(6)
    bar.setTextVisible(False)
    bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {bar_color}; }}")
    layout.addWidget(bar)
    return val_label, bar
