"""
ToolStrip — persistent three-row header bar.

Row 1 (Actions):   Title | Run  Reset  ▶Play | Speed | [frame slider] | Frame# | ⤢ Reset View
Row 2 (Vectors):   ☑ Force Vectors   [scale slider  0.1×–100×]  value
Row 3 (Ellips):    ☑ Mobility Ellipsoids [scale] value  |  ☑ Force Ellipsoids [scale] value

Design by Contract
------------------
Stateless: emits signals only; does not own simulation data.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Stylesheet constants
# ---------------------------------------------------------------------------

_STYLE_STRIP = "QWidget#toolstrip {background: #16162e;border-bottom: 1px solid #2a2a50;}"
_BTN_RUN = (
    "QPushButton{"
    "background:#1e5c30;color:#a8f0b8;border:none;border-radius:5px;"
    "padding:3px 12px;font-weight:bold;font-size:11px;"
    "}"
    "QPushButton:hover{background:#28784a;}"
    "QPushButton:pressed{background:#133d20;}"
    "QPushButton:disabled{background:#243424;color:#507050;}"
)
_BTN_RESET = (
    "QPushButton{"
    "background:#50282a;color:#f0b0b0;border:none;border-radius:5px;"
    "padding:3px 9px;font-size:10px;"
    "}"
    "QPushButton:hover{background:#6a3838;}"
    "QPushButton:disabled{background:#332020;color:#604040;}"
)
_BTN_PLAY = (
    "QPushButton{"
    "background:#1e2a50;color:#b0bce0;border:1px solid #303060;"
    "border-radius:5px;padding:3px 9px;font-size:10px;"
    "}"
    "QPushButton:checked{background:#50402a;color:#f0d070;border-color:#706030;}"
    "QPushButton:hover{background:#282860;}"
)
_BTN_SMALL = (
    "QPushButton{background:#1e2440;color:#9090c0;border:1px solid #303060;"
    "border-radius:4px;padding:2px 7px;font-size:9px;}"
    "QPushButton:hover{background:#252860;color:#b0b0e0;}"
)
_CHK_FORCE = (
    "QCheckBox{color:#a0e0b0;font-size:9px;spacing:3px;}"
    "QCheckBox::indicator{width:12px;height:12px;border:1px solid #405068;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#285040;border-color:#60a060;}"
)
_CHK_MOB = (
    "QCheckBox{color:#80c8f0;font-size:9px;spacing:3px;}"
    "QCheckBox::indicator{width:12px;height:12px;border:1px solid #304060;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#204060;border-color:#4080c0;}"
)
_CHK_FELL = (
    "QCheckBox{color:#f0b880;font-size:9px;spacing:3px;}"
    "QCheckBox::indicator{width:12px;height:12px;border:1px solid #604020;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#604020;border-color:#c08040;}"
)
_SLIDER_FORCE = (
    "QSlider::groove:horizontal{height:3px;background:#203028;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#60b070;border:none;"
    "width:10px;height:10px;margin:-4px 0;border-radius:5px;}"
    "QSlider::sub-page:horizontal{background:#406850;border-radius:2px;}"
)
_SLIDER_MOB = (
    "QSlider::groove:horizontal{height:3px;background:#202840;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#5898d0;border:none;"
    "width:10px;height:10px;margin:-4px 0;border-radius:5px;}"
    "QSlider::sub-page:horizontal{background:#305880;border-radius:2px;}"
)
_SLIDER_FELL = (
    "QSlider::groove:horizontal{height:3px;background:#402814;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#d08848;border:none;"
    "width:10px;height:10px;margin:-4px 0;border-radius:5px;}"
    "QSlider::sub-page:horizontal{background:#805028;border-radius:2px;}"
)
_SLIDER_FRAME = (
    "QSlider::groove:horizontal{height:3px;background:#202038;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#5858a8;border:none;"
    "width:10px;height:10px;margin:-4px 0;border-radius:5px;}"
    "QSlider::sub-page:horizontal{background:#383880;border-radius:2px;}"
)
_SEP_STYLE = "QFrame{color:#2a2a50;border:none;}"
_SEP_H_STYLE = "QFrame{color:#2a2a50;border:none;max-height:1px;}"
_LABEL = "color:#606080;font-size:9px;"
_VAL_LBL = "color:#8080b0;font-size:9px;font-family:monospace;min-width:32px;"
_FRAME_LBL = "color:#6060a0;font-size:9px;font-family:monospace;"
_TITLE = "color:#9090c8;font-size:11px;font-weight:bold;letter-spacing:1px;padding-right:4px;"


def _vline() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFixedWidth(1)
    sep.setStyleSheet(_SEP_STYLE)
    return sep


def _hline() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFixedHeight(1)
    sep.setStyleSheet(_SEP_H_STYLE)
    return sep


def _make_scale_slider(style: str, default: int = 10, max_val: int = 1000) -> QSlider:
    """Create a compact scale slider (1–max_val, default=10 → 1.0×).

    max_val=1000 → 0.1×…100× (force vectors, which can be very large)
    max_val=100  → 0.1×…10×  (ellipsoids, more subtle visual scaling)
    """
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(1, max_val)
    s.setValue(default)
    s.setStyleSheet(style)
    s.setFixedHeight(14)
    s.setMaximumWidth(200)
    return s


def _fmt_scale(raw: int) -> str:
    v = raw / 10.0
    return f"{v:.0f}×" if v >= 10 else f"{v:.1f}×"


class ToolStrip(QWidget):
    """Persistent three-row header toolbar.

    Row 1: simulation actions + playback.
    Row 2: Force Vectors checkbox + scale slider.
    Row 3: Mobility Ellipsoids + Force Ellipsoids (each with checkbox + scale).
    """

    # Simulation lifecycle
    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    frame_scrubbed = pyqtSignal(int)

    # Overlay toggles
    forces_toggled = pyqtSignal(bool)
    zero_torque_toggled = pyqtSignal(bool)
    mob_ellipsoid_toggled = pyqtSignal(bool)
    force_ellipsoid_toggled = pyqtSignal(bool)

    # Scale changes
    force_scale_changed = pyqtSignal(float)
    mob_scale_changed = pyqtSignal(float)
    force_ell_scale_changed = pyqtSignal(float)

    # View controls
    reset_view_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolstrip")
        self.setFixedHeight(105)
        self.setStyleSheet(_STYLE_STRIP)
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 2)
        outer.setSpacing(1)

        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self._build_row1(row1)
        outer.addLayout(row1)

        outer.addWidget(_hline())

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self._build_row2(row2)
        outer.addLayout(row2)

        outer.addWidget(_hline())

        row3 = QHBoxLayout()
        row3.setSpacing(4)
        self._build_row3(row3)
        outer.addLayout(row3)

    def _build_row1(self, layout: QHBoxLayout) -> None:
        """Actions row: Title | Run Reset Play | Speed | [frame slider] | Frame# | Reset View"""

        title = QLabel("Pendulums")
        title.setStyleSheet(_TITLE)
        title.setFont(QFont("Sans", 11, QFont.Weight.Bold))
        layout.addWidget(title)
        layout.addWidget(_vline())

        # Run / Reset / Play
        self.btn_run = QPushButton("▶ Run")
        self.btn_run.setStyleSheet(_BTN_RUN)
        self.btn_run.setToolTip("Run simulation")
        self.btn_run.clicked.connect(self.run_requested.emit)
        layout.addWidget(self.btn_run)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setStyleSheet(_BTN_RESET)
        self.btn_reset.setToolTip("Reset to initial state")
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self.btn_reset)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setStyleSheet(_BTN_PLAY)
        self.btn_play.setToolTip("Play / Pause animation")
        self.btn_play.toggled.connect(self._on_play_toggled)
        layout.addWidget(self.btn_play)

        layout.addWidget(_vline())

        # Speed
        spd_lbl = QLabel("Speed:")
        spd_lbl.setStyleSheet(_LABEL)
        layout.addWidget(spd_lbl)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.05, 20.0)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setFixedWidth(58)
        self.speed_spin.setStyleSheet(
            "QDoubleSpinBox{background:#1a1a2e;color:#b0b0d8;"
            "border:1px solid #303050;border-radius:4px;padding:1px 2px;"
            "font-size:10px;}"
        )
        self.speed_spin.setToolTip("Playback speed multiplier (0.05× – 20×)")
        self.speed_spin.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        layout.addWidget(self.speed_spin)

        layout.addWidget(_vline())

        # Playback frame slider
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setStyleSheet(_SLIDER_FRAME)
        self._frame_slider.setToolTip("Scrub to any frame")
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        layout.addWidget(self._frame_slider, stretch=1)

        self._frame_lbl = QLabel("0 / 0")
        self._frame_lbl.setStyleSheet(_FRAME_LBL)
        self._frame_lbl.setFixedWidth(58)
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._frame_lbl)

        layout.addWidget(_vline())

        # Reset View
        self.btn_reset_view = QPushButton("⤢ Reset View")
        self.btn_reset_view.setStyleSheet(_BTN_SMALL)
        self.btn_reset_view.setToolTip(
            "Reset zoom & pan to default\n(shortcut: double-click canvas)"
        )
        self.btn_reset_view.clicked.connect(self.reset_view_requested.emit)
        layout.addWidget(self.btn_reset_view)

    def _build_row2(self, layout: QHBoxLayout) -> None:
        """Force overlay row: ☑ Force Vectors [scale] | ☑ Zero-τ Forces"""

        self.chk_forces = QCheckBox("Force Vectors")
        self.chk_forces.setStyleSheet(_CHK_FORCE)
        self.chk_forces.setFixedWidth(100)
        self.chk_forces.setToolTip(
            "Show net joint force vectors at each joint.\n"
            "Arrow length scales with force magnitude."
        )
        self.chk_forces.toggled.connect(self.forces_toggled.emit)
        layout.addWidget(self.chk_forces)

        self._sld_force = _make_scale_slider(_SLIDER_FORCE, default=10)
        self._sld_force.setToolTip("Force vector display scale (0.1× – 100×)")
        self._sld_force.valueChanged.connect(self._on_force_scale)
        layout.addWidget(self._sld_force)

        self._lbl_force_scale = QLabel("1.0×")
        self._lbl_force_scale.setStyleSheet(_VAL_LBL)
        layout.addWidget(self._lbl_force_scale)

        layout.addWidget(_vline())

        self.chk_zero_torque = QCheckBox("Zero-τ Forces")
        self.chk_zero_torque.setStyleSheet(
            "QCheckBox{color:#d0a0e0;font-size:9px;spacing:3px;}"
            "QCheckBox::indicator{width:12px;height:12px;border:1px solid #604080;"
            "border-radius:3px;background:#1a1a2a;}"
            "QCheckBox::indicator:checked{background:#602080;border-color:#a060c0;}"
        )
        self.chk_zero_torque.setToolTip(
            "Show zero-torque counterfactual forces (dashed vectors).\n"
            "These represent joint forces if all driving torques were removed—\n"
            "the passive drift due to gravity and inertia alone."
        )
        self.chk_zero_torque.toggled.connect(self.zero_torque_toggled.emit)
        layout.addWidget(self.chk_zero_torque)

        layout.addStretch()

    def _build_row3(self, layout: QHBoxLayout) -> None:
        """Ellipsoid row: ☑ Mobility [scale] value | ☑ Force Ell [scale] value | [status]"""

        self.chk_mob = QCheckBox("Mobility Ellipsoids")
        self.chk_mob.setStyleSheet(_CHK_MOB)
        self.chk_mob.setFixedWidth(130)
        self.chk_mob.setToolTip(
            "Show mobility ellipsoids at segment endpoints.\n"
            "Cyan = achievable velocity; large = high dexterity."
        )
        self.chk_mob.toggled.connect(self.mob_ellipsoid_toggled.emit)
        layout.addWidget(self.chk_mob)

        self._sld_mob = _make_scale_slider(_SLIDER_MOB, default=10, max_val=100)
        self._sld_mob.setToolTip("Mobility ellipsoid display scale (0.1× – 10×)")
        self._sld_mob.valueChanged.connect(self._on_mob_scale)
        layout.addWidget(self._sld_mob, stretch=1)

        self._lbl_mob_scale = QLabel("1.0×")
        self._lbl_mob_scale.setStyleSheet(_VAL_LBL)
        layout.addWidget(self._lbl_mob_scale)

        layout.addWidget(_vline())

        self.chk_force_ell = QCheckBox("Force Ellipsoids")
        self.chk_force_ell.setStyleSheet(_CHK_FELL)
        self.chk_force_ell.setFixedWidth(110)
        self.chk_force_ell.setToolTip(
            "Show force ellipsoids at segment endpoints.\n"
            "Orange = achievable endpoint force; small = near-singular."
        )
        self.chk_force_ell.toggled.connect(self.force_ellipsoid_toggled.emit)
        layout.addWidget(self.chk_force_ell)

        self._sld_force_ell = _make_scale_slider(_SLIDER_FELL, default=10, max_val=100)
        self._sld_force_ell.setToolTip("Force ellipsoid display scale (0.1× – 10×)")
        self._sld_force_ell.valueChanged.connect(self._on_force_ell_scale)
        layout.addWidget(self._sld_force_ell, stretch=1)

        self._lbl_force_ell_scale = QLabel("1.0×")
        self._lbl_force_ell_scale.setStyleSheet(_VAL_LBL)
        layout.addWidget(self._lbl_force_ell_scale)

        layout.addWidget(_vline())

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color:#404060;font-size:9px;")
        layout.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Slots / public API
    # ------------------------------------------------------------------

    def _on_play_toggled(self, checked: bool) -> None:
        self.btn_play.setText("⏸ Pause" if checked else "▶ Play")
        self.play_toggled.emit(checked)

    def _on_frame_slider_changed(self, val: int) -> None:
        total = self._frame_slider.maximum()
        self._frame_lbl.setText(f"{val} / {total}")
        self.frame_scrubbed.emit(val)

    def _on_force_scale(self, raw: int) -> None:
        self._lbl_force_scale.setText(_fmt_scale(raw))
        self.force_scale_changed.emit(raw / 10.0)

    def _on_mob_scale(self, raw: int) -> None:
        self._lbl_mob_scale.setText(_fmt_scale(raw))
        self.mob_scale_changed.emit(raw / 10.0)

    def _on_force_ell_scale(self, raw: int) -> None:
        self._lbl_force_ell_scale.setText(_fmt_scale(raw))
        self.force_ell_scale_changed.emit(raw / 10.0)

    def set_status(self, msg: str) -> None:
        """Update the right-hand status label."""
        self._status_lbl.setText(msg)

    def set_running(self, running: bool) -> None:
        """Disable run/reset while simulation is computing."""
        self.btn_run.setEnabled(not running)
        self.btn_reset.setEnabled(not running)
        self.set_status("Simulating…" if running else "Ready")

    def set_frame_range(self, n_steps: int) -> None:
        """Set the playback slider maximum after simulation completes."""
        assert n_steps >= 0
        self._frame_slider.setRange(0, max(0, n_steps - 1))
        self._frame_slider.setValue(0)
        total = max(0, n_steps - 1)
        self._frame_lbl.setText(f"0 / {total}")

    def set_frame(self, idx: int) -> None:
        """Update slider + label to reflect current frame (no re-emission)."""
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)
        total = self._frame_slider.maximum()
        self._frame_lbl.setText(f"{idx} / {total}")

    def stop_play(self) -> None:
        """Force the Play button to the uncheck (stopped) state."""
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶ Play")
        self.btn_play.blockSignals(False)
