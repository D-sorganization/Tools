"""
ToolStrip — persistent two-row header bar with the most important controls.

Row 1 (Actions):  Title | Run  Reset  Play/Pause | Speed | Playback slider | Frame#
Row 2 (Overlays): Force Vectors | Mobility Ellipsoids | Force Ellipsoids | [status]

Always visible above the tab widget regardless of scroll position.

Signals
-------
run_requested       : user clicked Run
reset_requested     : user clicked Reset
play_toggled(bool)  : Play/Pause toggled
speed_changed(float): Speed spin changed
frame_scrubbed(int) : User dragged the playback slider
forces_toggled(bool): Force vectors checkbox toggled
mob_ellipsoid_toggled(bool)  : Mobility ellipsoid checkbox toggled
force_ellipsoid_toggled(bool): Force ellipsoid checkbox toggled

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

_STYLE_STRIP = (
    "QWidget#toolstrip {" "background: #16162e;" "border-bottom: 1px solid #2a2a50;" "}"
)
_BTN_RUN = (
    "QPushButton{"
    "background:#1e5c30;color:#a8f0b8;border:none;border-radius:5px;"
    "padding:4px 14px;font-weight:bold;font-size:11px;"
    "}"
    "QPushButton:hover{background:#28784a;}"
    "QPushButton:pressed{background:#133d20;}"
    "QPushButton:disabled{background:#243424;color:#507050;}"
)
_BTN_RESET = (
    "QPushButton{"
    "background:#50282a;color:#f0b0b0;border:none;border-radius:5px;"
    "padding:4px 10px;font-size:10px;"
    "}"
    "QPushButton:hover{background:#6a3838;}"
    "QPushButton:disabled{background:#332020;color:#604040;}"
)
_BTN_PLAY = (
    "QPushButton{"
    "background:#1e2a50;color:#b0bce0;border:1px solid #303060;"
    "border-radius:5px;padding:4px 10px;font-size:10px;"
    "}"
    "QPushButton:checked{background:#50402a;color:#f0d070;border-color:#706030;}"
    "QPushButton:hover{background:#282860;}"
)
_CHK = (
    "QCheckBox{color:#9090b8;font-size:9px;spacing:3px;}"
    "QCheckBox::indicator{width:12px;height:12px;border:1px solid #404068;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#406840;border-color:#60a060;}"
)
_CHK_FORCE_VEC = (
    "QCheckBox{color:#a0d8b0;font-size:9px;spacing:3px;}"
    "QCheckBox::indicator{width:12px;height:12px;border:1px solid #405068;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#285040;border-color:#60a060;}"
)
_SLIDER = (
    "QSlider::groove:horizontal{height:4px;background:#252538;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#6060a0;border:none;"
    "width:12px;height:12px;margin:-4px 0;border-radius:6px;}"
    "QSlider::sub-page:horizontal{background:#404080;border-radius:2px;}"
)
_SEP_STYLE = "QFrame{color:#2a2a50;border:none;}"
_LABEL = "color:#606080;font-size:9px;"
_FRAME_LBL = "color:#6060a0;font-size:9px;font-family:monospace;"


def _vline() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFixedWidth(1)
    sep.setStyleSheet(_SEP_STYLE)
    return sep


class ToolStrip(QWidget):
    """Persistent two-row header toolbar.

    Row 1: simulation actions + playback slider.
    Row 2: display overlay toggles.
    """

    # Simulation lifecycle
    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    frame_scrubbed = pyqtSignal(int)

    # Overlay toggles
    forces_toggled = pyqtSignal(bool)
    mob_ellipsoid_toggled = pyqtSignal(bool)
    force_ellipsoid_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolstrip")
        self.setFixedHeight(60)  # two rows
        self.setStyleSheet(_STYLE_STRIP)
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 2, 8, 2)
        outer.setSpacing(1)

        row1 = QHBoxLayout()
        row1.setSpacing(5)
        row2 = QHBoxLayout()
        row2.setSpacing(5)

        self._build_row1(row1)
        self._build_row2(row2)

        outer.addLayout(row1)
        outer.addLayout(row2)

    def _build_row1(self, layout: QHBoxLayout) -> None:
        """Actions row: Title | Run Reset Play | Speed | Slider | Frame"""

        # Title
        title = QLabel("Pendulums")
        title.setFont(QFont("Sans", 10, QFont.Weight.Bold))
        title.setStyleSheet("color:#9090d8;")
        layout.addWidget(title)
        layout.addWidget(_vline())

        # Buttons
        self.btn_run = QPushButton("▶  Run")
        self.btn_run.setStyleSheet(_BTN_RUN)
        self.btn_run.setToolTip("Run simulation (Ctrl+R)")
        self.btn_run.clicked.connect(self.run_requested.emit)
        layout.addWidget(self.btn_run)

        self.btn_reset = QPushButton("⟳ Reset")
        self.btn_reset.setStyleSheet(_BTN_RESET)
        self.btn_reset.setToolTip("Reset simulation")
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
        self.speed_spin.setRange(0.1, 10.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setFixedWidth(55)
        self.speed_spin.setSuffix("×")
        self.speed_spin.setToolTip("Playback speed multiplier")
        self.speed_spin.setStyleSheet(
            "background:#1e1e30;color:#e0e0f0;border:1px solid #404060;"
            "border-radius:3px;padding:1px 3px;font-size:9px;"
        )
        self.speed_spin.valueChanged.connect(
            lambda v: self.speed_changed.emit(float(v))
        )
        layout.addWidget(self.speed_spin)

        layout.addWidget(_vline())

        # Playback slider
        pb_lbl = QLabel("Frame:")
        pb_lbl.setStyleSheet(_LABEL)
        layout.addWidget(pb_lbl)

        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.setMinimum(0)
        self.playback_slider.setMaximum(0)
        self.playback_slider.setValue(0)
        self.playback_slider.setMinimumWidth(120)
        self.playback_slider.setMaximumWidth(300)
        self.playback_slider.setToolTip("Scrub through simulation frames")
        self.playback_slider.setStyleSheet(_SLIDER)
        self.playback_slider.valueChanged.connect(lambda v: self.frame_scrubbed.emit(v))
        layout.addWidget(self.playback_slider, stretch=1)

        self._frame_lbl = QLabel("0 / 0")
        self._frame_lbl.setStyleSheet(_FRAME_LBL)
        self._frame_lbl.setFixedWidth(65)
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._frame_lbl)

    def _build_row2(self, layout: QHBoxLayout) -> None:
        """Overlay toggles row: Force Vectors | Mob Ellipsoids | Force Ellipsoids | [status]"""

        ovl_lbl = QLabel("Show:")
        ovl_lbl.setStyleSheet(_LABEL)
        layout.addWidget(ovl_lbl)

        # Force vectors
        self.chk_forces = QCheckBox("Force Vectors")
        self.chk_forces.setStyleSheet(_CHK_FORCE_VEC)
        self.chk_forces.setToolTip(
            "Show net joint force vectors at each joint.\n"
            "Arrow length scaled by force magnitude."
        )
        self.chk_forces.toggled.connect(self.forces_toggled.emit)
        layout.addWidget(self.chk_forces)

        layout.addWidget(_vline())

        ell_lbl = QLabel("Ellipsoids:")
        ell_lbl.setStyleSheet(_LABEL)
        layout.addWidget(ell_lbl)

        self.chk_mob = QCheckBox("Mobility")
        self.chk_mob.setStyleSheet(_CHK)
        self.chk_mob.setToolTip(
            "Show mobility ellipsoids at each segment endpoint.\n"
            "Cyan ellipse = achievable velocity range for unit joint speed.\n"
            "Large ellipse = high dexterity."
        )
        self.chk_mob.toggled.connect(self.mob_ellipsoid_toggled.emit)
        layout.addWidget(self.chk_mob)

        self.chk_force_ell = QCheckBox("Force")
        self.chk_force_ell.setStyleSheet(_CHK)
        self.chk_force_ell.setToolTip(
            "Show force ellipsoids at each segment endpoint.\n"
            "Orange ellipse = achievable endpoint force for unit joint torque.\n"
            "Small = near singular; large = good force transmission."
        )
        self.chk_force_ell.toggled.connect(self.force_ellipsoid_toggled.emit)
        layout.addWidget(self.chk_force_ell)

        layout.addStretch()

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color:#505070;font-size:9px;")
        layout.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Slots / public API
    # ------------------------------------------------------------------

    def _on_play_toggled(self, checked: bool) -> None:
        self.btn_play.setText("⏸ Pause" if checked else "▶ Play")
        self.play_toggled.emit(checked)

    def set_status(self, msg: str) -> None:
        """Update the right-hand status label."""
        self._status_lbl.setText(msg)

    def set_running(self, running: bool) -> None:
        """Disable run/reset while simulation is computing."""
        self.btn_run.setEnabled(not running)
        self.btn_reset.setEnabled(not running)
        self._status_lbl.setText("⏳ Simulating…" if running else "Ready")

    def set_frame_range(self, n_steps: int) -> None:
        """Update slider range after a new simulation is loaded."""
        assert n_steps >= 0, f"n_steps must be non-negative, got {n_steps}"
        self.playback_slider.setMaximum(max(0, n_steps - 1))
        self._update_frame_label(0, n_steps)

    def set_frame(self, idx: int) -> None:
        """Move slider to frame index without emitting frame_scrubbed."""
        self.playback_slider.blockSignals(True)
        self.playback_slider.setValue(idx)
        self.playback_slider.blockSignals(False)
        n = self.playback_slider.maximum() + 1
        self._update_frame_label(idx, n)

    def _update_frame_label(self, idx: int, n_steps: int) -> None:
        self._frame_lbl.setText(f"{idx} / {max(0, n_steps - 1)}")

    def stop_play(self) -> None:
        """Programmatically stop playback without emitting play_toggled."""
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶ Play")
        self.btn_play.blockSignals(False)
