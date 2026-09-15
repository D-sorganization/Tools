"""The putting monitor's window: live view, readout, GSPro, calibration.

:class:`PuttingMonitorWindow` loads the per-user calibration, runs
:class:`PuttingMonitor` on a worker thread and shows what it reports. Every
measurement on screen came out of the monitor; the window only draws,
starts, stops and edits the calibration through the wizard and the HSV
tuner. Camera, snapshot and GSPro client are injectable so the window is
built and driven in tests without hardware.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from putting_launch_monitor.calibration import Calibration, default_calibration_path
from putting_launch_monitor.cli import scaled_calibration
from putting_launch_monitor.detect import Frame, HsvRange
from putting_launch_monitor.monitor import FrameEvent
from shared.python.contracts import StateError
from shared.python.theme import ThemedWindowMixin

from .calibration_wizard import CalibrationWizard, SnapshotFn
from .gspro_panel import ClientFactory, GsproPanel, default_client
from .hsv_tuner import HsvTunerDialog
from .live_view import LiveView
from .painting import TOKENS
from .readout import PuttReadout
from .worker import (
    DEFAULT_DECODE_WIDTH,
    EventBridge,
    MonitorWorker,
    SourceFactory,
    build_monitor,
    camera_source,
    take_snapshot,
)

logger = logging.getLogger(__name__)

SETTINGS_APP = "PuttingLaunchMonitor"
MIN_DECODE_WIDTH, MAX_DECODE_WIDTH = 160, 4096
_DIALOG_ACCEPTED = QDialog.DialogCode.Accepted


class PuttingMonitorWindow(ThemedWindowMixin, QMainWindow):
    """Live view on the left; controls, putt readout and GSPro on the right."""

    def __init__(
        self,
        calibration_path: Path | None = None,
        *,
        source_factory: SourceFactory = camera_source,
        snapshot: SnapshotFn = take_snapshot,
        client_factory: ClientFactory = default_client,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.calibration_path = calibration_path or default_calibration_path()
        self._source_factory = source_factory
        self._snapshot = snapshot
        self.calibration: Calibration | None = None
        self.worker: MonitorWorker | None = None
        self.bridge: EventBridge | None = None
        self.last_event: FrameEvent | None = None
        self._last_frame: Frame | None = None
        self.setWindowTitle("Putting Launch Monitor")
        self.live_view = LiveView()
        self.readout = PuttReadout()
        self.gspro = GsproPanel(client_factory)
        self._build_controls()
        self._layout()
        self.setup_theme_support(add_menu=True, settings_app=SETTINGS_APP)
        self.load_calibration()

    # -- construction ---------------------------------------------------------------
    def _build_controls(self) -> None:
        self.controls = QGroupBox("Monitor")
        self.camera_id = QLineEdit()
        self.camera_id.setToolTip(
            "DirectShow camera by PnP instance id (see the handoff for the "
            "rig's ids). Saved into the calibration when the monitor starts."
        )
        self.decode_width = QSpinBox()
        self.decode_width.setRange(MIN_DECODE_WIDTH, MAX_DECODE_WIDTH)
        self.decode_width.setValue(DEFAULT_DECODE_WIDTH)
        self.decode_width.setToolTip(
            "Width the stream is decoded at; smaller is faster, larger is "
            "more precise. The calibration is rescaled to match."
        )
        self.start_button = QPushButton("Start")
        self.start_button.setToolTip("Open the camera and watch for putts.")
        self.start_button.clicked.connect(self.toggle)
        self.calibrate_button = QPushButton("Calibrate…")
        self.calibrate_button.setToolTip(
            "Snapshot, click the mat corners, enter its size; stops the monitor."
        )
        self.calibrate_button.clicked.connect(self.run_wizard)
        self.tune_button = QPushButton("Tune colour…")
        self.tune_button.setToolTip(
            "Adjust the ball's HSV bounds against the last frame, mask shown."
        )
        self.tune_button.clicked.connect(self.run_tuner)
        self.calibration_label = QLabel()
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setToolTip("Where the calibration lives.")
        buttons = QHBoxLayout()
        buttons.setSpacing(TOKENS.spacing_px)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.calibrate_button)
        buttons.addWidget(self.tune_button)
        column = QVBoxLayout(self.controls)
        column.setSpacing(TOKENS.spacing_px)
        column.addWidget(QLabel("Camera id"))
        column.addWidget(self.camera_id)
        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Decode width (px)"))
        width_row.addWidget(self.decode_width, 1)
        column.addLayout(width_row)
        column.addLayout(buttons)
        column.addWidget(self.calibration_label)

    def _layout(self) -> None:
        side = QVBoxLayout()
        side.setSpacing(TOKENS.padding_px)
        side.addWidget(self.controls)
        side.addWidget(self.readout)
        side.addWidget(self.gspro)
        side.addStretch()
        central = QWidget()
        row = QHBoxLayout(central)
        pad = TOKENS.padding_px
        row.setContentsMargins(pad, pad, pad, pad)
        row.setSpacing(pad)
        row.addWidget(self.live_view, 1)
        row.addLayout(side)
        self.setCentralWidget(central)
        self._message = ""
        self.say("ready")

    def say(self, message: str) -> None:
        """Show ``message`` in the status bar (and remember it)."""
        self._message = message
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(message)

    # -- calibration ------------------------------------------------------------------
    def load_calibration(self) -> None:
        """Adopt the calibration at ``calibration_path`` if there is one."""
        if self.calibration_path.is_file():
            try:
                self.calibration = Calibration.load(self.calibration_path)
            except (ValueError, OSError) as exc:
                logger.warning("calibration unreadable: %s", exc)
                self.calibration = None
                self.say(f"calibration unreadable: {exc}")
        else:
            self.calibration = None
        self._show_calibration()

    def _show_calibration(self) -> None:
        cal = self.calibration
        if cal is None:
            self.calibration_label.setText(
                f"No calibration at {self.calibration_path}. Run Calibrate first."
            )
            self.live_view.set_region(None)
            self.start_button.setEnabled(False)
            self.tune_button.setEnabled(False)
            return
        self.camera_id.setText(cal.camera_instance_id)
        self.calibration_label.setText(
            f"{self.calibration_path.name}: mat {cal.mat_width_mm:.0f} x "
            f"{cal.mat_length_mm:.0f} mm, {cal.colour} ball, "
            f"{cal.capture_width}x{cal.capture_height}@{cal.fps}"
        )
        self.live_view.set_region(self._scaled().search_region())
        self.start_button.setEnabled(True)
        self.tune_button.setEnabled(True)

    def current(self) -> Calibration:
        """The loaded calibration. Precondition: one is loaded."""
        if self.calibration is None:
            raise StateError("no calibration is loaded")
        return self.calibration

    def _scaled(self) -> Calibration:
        cal = self._with_camera_field(self.current())
        return scaled_calibration(cal, self.decode_width.value())

    def _with_camera_field(self, cal: Calibration) -> Calibration:
        text = self.camera_id.text().strip()
        return replace(cal, camera_instance_id=text) if text else cal

    def run_wizard(self) -> None:
        """Stop the monitor (the camera is single-open), then calibrate."""
        self.stop()
        wizard = CalibrationWizard(
            self.calibration_path,
            initial=self.calibration,
            snapshot=self._snapshot,
            parent=self,
        )
        if wizard.exec() == _DIALOG_ACCEPTED:
            self.load_calibration()
            self.say(f"saved {self.calibration_path}")

    def run_tuner(self) -> None:
        """Tune the HSV bounds against the last frame; OK saves them."""
        if self.calibration is None:
            return
        dialog = HsvTunerDialog(self._scaled().detector(), self._last_frame, self)
        if dialog.exec() == _DIALOG_ACCEPTED:
            self.apply_colour(dialog.hsv_range())

    def apply_colour(self, colour: HsvRange) -> None:
        """Persist a custom HSV range into the calibration and reload it."""
        updated = replace(self.current(), custom_colour=colour)
        updated.save(self.calibration_path)
        self.load_calibration()
        self.say("ball colour saved; restart the monitor to apply")

    # -- running --------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        """Build the monitor over the camera and run it on the worker thread."""
        if self.running or self.calibration is None:
            return
        cal = self._with_camera_field(self.calibration)
        if cal != self.calibration:
            cal.save(self.calibration_path)
            self.calibration = cal
        self.bridge = EventBridge(self)
        self.bridge.arrived.connect(self.on_event)
        try:
            monitor = build_monitor(
                cal,
                self.decode_width.value(),
                self.gspro.sink_for_monitor(),
                self.bridge,
                self._source_factory,
            )
        except (ValueError, StateError, OSError) as exc:
            self.say(f"cannot start: {exc}")
            return
        self.live_view.set_region(monitor.calibration.search_region())
        self.readout.clear()
        self.worker = MonitorWorker(monitor, self)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._set_running(True)
        self.say("watching for a resting ball")

    def stop(self) -> None:
        if self.worker is None:
            return
        if not self.worker.stop():
            logger.warning("monitor thread did not stop in time")
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.start_button.setText("Stop" if running else "Start")
        for widget in (self.calibrate_button, self.decode_width, self.camera_id):
            widget.setEnabled(not running)

    def _on_failed(self, message: str) -> None:
        self.say(f"monitor stopped: {message}")

    def _on_finished(self) -> None:
        self._set_running(False)
        if self._message == "watching for a resting ball":
            self.say("monitor stopped")

    def on_event(self, event: FrameEvent) -> None:
        """One :class:`FrameEvent`, on the GUI thread."""
        self.last_event = event
        if event.frame is not None:
            self._last_frame = event.frame
        self.live_view.show_event(event)
        self.readout.show_event(event)
        if event.putt is not None:
            self.gspro.refresh()
            self.say(event.outcome or event.putt.reason)
        if self.bridge is not None:
            self.bridge.release()

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        self.stop()
        self.gspro.close_connection()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    """Run the window; the theme mixin applies the fleet theme."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    app = QApplication(argv if argv is not None else sys.argv)
    window = PuttingMonitorWindow()
    window.resize(1280, 800)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
