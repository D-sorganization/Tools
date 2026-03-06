"""
ToolStrip — persistent header bar with the most important simulation controls.

Why a separate widget?
  The controls panel is scrollable and can be partially off-screen at small
  window heights.  The toolstrip is always visible at the top of the window,
  giving one-click access to the most-used actions.

Signals forwarded
-----------------
The toolstrip duplicates a minimal set of the ControlsWidget signals so that
SimulationPanel can connect to either source independently:

    run_requested       → same as ControlsWidget.run_requested
    reset_requested     → same as ControlsWidget.reset_requested
    play_toggled(bool)  → same as ControlsWidget.play_toggled
    speed_changed(float)→ same as ControlsWidget.speed_changed

Ellipsoid toggles (new, not in ControlsWidget):
    mob_ellipsoid_toggled(bool)   → show/hide mobility ellipsoids
    force_ellipsoid_toggled(bool) → show/hide force ellipsoids

Design by Contract
------------------
The toolstrip is stateless — it emits signals and does not cache simulation
results.  The parent (SimulationPanel) owns all state.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

_STYLE_STRIP = (
    "QWidget#toolstrip {" "background: #16162e;" "border-bottom: 1px solid #2a2a50;" "}"
)

_BTN_RUN = (
    "QPushButton{"
    "background:#1e5c30;color:#a8f0b8;border:none;border-radius:5px;"
    "padding:5px 16px;font-weight:bold;font-size:12px;"
    "}"
    "QPushButton:hover{background:#28784a;}"
    "QPushButton:pressed{background:#133d20;}"
    "QPushButton:disabled{background:#243424;color:#507050;}"
)
_BTN_RESET = (
    "QPushButton{"
    "background:#50282a;color:#f0b0b0;border:none;border-radius:5px;"
    "padding:5px 12px;font-size:11px;"
    "}"
    "QPushButton:hover{background:#6a3838;}"
)
_BTN_PLAY = (
    "QPushButton{"
    "background:#1e2a50;color:#b0bce0;border:1px solid #303060;"
    "border-radius:5px;padding:5px 12px;font-size:11px;"
    "}"
    "QPushButton:checked{background:#50402a;color:#f0d070;border-color:#706030;}"
    "QPushButton:hover{background:#282860;}"
)
_CHK = (
    "QCheckBox{color:#9090b8;font-size:10px;spacing:4px;}"
    "QCheckBox::indicator{width:13px;height:13px;border:1px solid #404068;"
    "border-radius:3px;background:#1a1a2a;}"
    "QCheckBox::indicator:checked{background:#406840;border-color:#60a060;}"
)
_SEP_STYLE = "QFrame{color:#2a2a50;border:none;}"
_LABEL = "color:#606080;font-size:10px;"


def _vline() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFixedWidth(1)
    sep.setStyleSheet(_SEP_STYLE)
    return sep


class ToolStrip(QWidget):
    """Persistent header toolbar with essential simulation controls.

    Emits signals that parallel the ControlsWidget API so that
    SimulationPanel can connect to both interchangeably.
    """

    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    mob_ellipsoid_toggled = pyqtSignal(bool)
    force_ellipsoid_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolstrip")
        self.setFixedHeight(40)
        self.setStyleSheet(_STYLE_STRIP)
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)

        # ── Title ────────────────────────────────────────────────────
        title = QLabel("Pendulums")
        title.setFont(QFont("Sans", 11, QFont.Weight.Bold))
        title.setStyleSheet("color:#9090d8;")
        layout.addWidget(title)
        layout.addWidget(_vline())

        # ── Run / Reset / Play ────────────────────────────────────────
        self.btn_run = QPushButton("▶  Run")
        self.btn_run.setStyleSheet(_BTN_RUN)
        self.btn_run.setToolTip("Run simulation (Ctrl+R)")
        self.btn_run.clicked.connect(self.run_requested.emit)
        layout.addWidget(self.btn_run)

        self.btn_reset = QPushButton("⟳ Reset")
        self.btn_reset.setStyleSheet(_BTN_RESET)
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self.btn_reset)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setStyleSheet(_BTN_PLAY)
        self.btn_play.setToolTip("Play / Pause animation")
        self.btn_play.toggled.connect(self._on_play_toggled)
        layout.addWidget(self.btn_play)

        layout.addWidget(_vline())

        # ── Speed ─────────────────────────────────────────────────────
        spd_lbl = QLabel("Speed:")
        spd_lbl.setStyleSheet(_LABEL)
        layout.addWidget(spd_lbl)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 10.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setFixedWidth(58)
        self.speed_spin.setSuffix("×")
        self.speed_spin.setToolTip("Playback speed multiplier")
        self.speed_spin.setStyleSheet(
            "background:#1e1e30;color:#e0e0f0;border:1px solid #404060;"
            "border-radius:3px;padding:1px 3px;font-size:10px;"
        )
        self.speed_spin.valueChanged.connect(lambda v: self.speed_changed.emit(v))
        layout.addWidget(self.speed_spin)

        layout.addWidget(_vline())

        # ── Ellipsoid toggles ─────────────────────────────────────────
        ell_lbl = QLabel("Ellipsoids:")
        ell_lbl.setStyleSheet(_LABEL)
        layout.addWidget(ell_lbl)

        self.chk_mob = QCheckBox("Mobility")
        self.chk_mob.setStyleSheet(_CHK)
        self.chk_mob.setToolTip(
            "Show mobility ellipsoids at each segment endpoint.\n"
            "Ellipse size = achievable velocity range for unit joint speed."
        )
        self.chk_mob.toggled.connect(self.mob_ellipsoid_toggled.emit)
        layout.addWidget(self.chk_mob)

        self.chk_force = QCheckBox("Force")
        self.chk_force.setStyleSheet(_CHK)
        self.chk_force.setToolTip(
            "Show force ellipsoids at each segment endpoint.\n"
            "Ellipse size = achievable endpoint force for unit joint torque."
        )
        self.chk_force.toggled.connect(self.force_ellipsoid_toggled.emit)
        layout.addWidget(self.chk_force)

        layout.addStretch()

        # ── Status label (right-aligned) ──────────────────────────────
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color:#505070;font-size:10px;")
        layout.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Slots
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

    def stop_play(self) -> None:
        """Programmatically stop playback without emitting play_toggled."""
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶ Play")
        self.btn_play.blockSignals(False)
