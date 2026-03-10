"""Main window assembling all panels for the Asteroid Jumper simulation."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QWidget,
)

from asteroid_jumper.controller import SimController
from asteroid_jumper.controls_panel import ControlsPanel
from asteroid_jumper.metrics_panel import MetricsPanel
from asteroid_jumper.renderer import AsteroidJumperRenderer


class AsteroidJumperWindow(QMainWindow):
    """Top-level window for the Asteroid Jumper simulation."""

    def __init__(self) -> None:
        super().__init__()
        self._ctrl = SimController()
        self._build_ui()
        self._connect_signals()
        self._start_metrics_timer()
        self.setWindowTitle("Asteroid Jumper — Rigid Body Physics Simulation")
        self.resize(1200, 700)

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # Left: controls
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(230)
        controls_scroll.setMaximumWidth(280)
        self._controls = ControlsPanel(self._ctrl)
        controls_scroll.setWidget(self._controls)
        splitter.addWidget(controls_scroll)

        # Centre: renderer
        self._renderer = AsteroidJumperRenderer(self._ctrl)
        splitter.addWidget(self._renderer)

        # Right: metrics
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setMinimumWidth(230)
        metrics_scroll.setMaximumWidth(280)
        self._metrics = MetricsPanel(self._ctrl)
        metrics_scroll.setWidget(self._metrics)
        splitter.addWidget(metrics_scroll)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        # Status bar
        self._status_msg(
            "Drag on the asteroid to set jump direction · Click JUMP to launch"
        )

    def _connect_signals(self) -> None:
        self._controls.jump_requested.connect(self._on_jump)
        self._controls.reset_requested.connect(self._on_reset)
        self._controls.config_changed.connect(self._renderer.update)
        # Attach lightweight signal for mouse-drag angle update
        self._renderer.force_angle_changed.connect(self._controls.set_force_angle)

    def _start_metrics_timer(self) -> None:
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(100)  # 10 Hz
        self._metrics_timer.timeout.connect(self._metrics.refresh)
        self._metrics_timer.start()

    def _status_msg(self, msg: str) -> None:
        """Show a message in the status bar (null-safe)."""
        sb = self.statusBar()
        if sb is not None:
            sb.showMessage(msg)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_jump(self) -> None:
        if self._ctrl.state.phase != "ready":
            return
        self._ctrl.start_jump()
        self._renderer.start_animation()
        self._controls.enable_controls(False)
        self._status_msg("Launching…  watch the spin!")

    def _on_reset(self) -> None:
        self._renderer.stop_animation()
        self._ctrl.reset()
        self._renderer.reset_view()
        self._controls.enable_controls(True)
        self._controls.sync_from_controller()
        self._status_msg(
            "Reset · Drag on the asteroid to set jump direction · Click JUMP to launch"
        )


# ---------------------------------------------------------------------------
# Mini signal helper — avoids importing full PyQt signal machinery
# ---------------------------------------------------------------------------


class _SimpleSignal:
    """Lightweight callable signal (wraps a list of callbacks)."""

    def __init__(self) -> None:
        self._slots: list[object] = []

    def connect(self, slot: object) -> None:
        assert callable(slot)
        self._slots.append(slot)

    def emit(self, *args: object) -> None:
        for slot in self._slots:
            slot(*args)  # type: ignore[operator]
