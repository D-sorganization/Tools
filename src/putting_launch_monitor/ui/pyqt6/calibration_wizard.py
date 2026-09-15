"""The calibration wizard: snapshot, corners, mat, review, save.

:class:`CalibrationWizard` walks the operator from a camera to a saved
:class:`Calibration`. The camera page takes a snapshot through the shared
camera source (or loads an image file when no camera is at hand); the
review page shows the reprojection error and the search region the
calibration implies, then :meth:`accept` saves through
:meth:`Calibration.save`. Snapshot taking is injectable so the wizard is
tested without a camera.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from putting_launch_monitor.calibration import Calibration
from putting_launch_monitor.detect import Frame, RegionOfInterest
from shared.python.camera import CaptureMode
from shared.python.contracts import StateError, require

from .live_view import CornerCanvas
from .painting import TOKENS
from .wizard_pages import CornersPage, MatPage, WizardState
from .worker import take_snapshot

logger = logging.getLogger(__name__)

SnapshotFn = Callable[[str, CaptureMode], Frame]
MAX_CAPTURE_PX = 8192
MAX_FPS = 240
_STYLE = QWizard.WizardStyle.ModernStyle


def calibration_from_state(state: WizardState) -> Calibration:
    """The calibration the wizard's state describes.

    Precondition: four corners clicked. Everything else has a default the
    :class:`Calibration` contract accepts.
    """
    require(len(state.corners) == 4, "four corners", len(state.corners))
    roi = RegionOfInterest(*state.roi) if state.roi is not None else None
    return Calibration(
        camera_instance_id=state.camera_instance_id,
        mat_corners_px=tuple(state.corners),
        mat_width_mm=state.mat_width_mm,
        mat_length_mm=state.mat_length_mm,
        target_deg=state.target_deg,
        colour=state.colour,
        custom_colour=state.custom_colour,
        roi=roi,
        fps=state.fps,
        capture_width=state.capture_width,
        capture_height=state.capture_height,
        notes=state.notes,
    )


def load_image_file(path: Path) -> Frame:
    """A BGR frame from an image file on disk (for calibrating from a saved shot)."""
    import cv2

    require(path.is_file(), "image file must exist", str(path))
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise StateError(f"could not decode {path}")
    return image


class CameraPage(QWizardPage):
    """Which camera, which mode, and a snapshot to click on."""

    def __init__(
        self,
        state: WizardState,
        snapshot: SnapshotFn,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._snapshot = snapshot
        self.setTitle("Camera")
        self.setSubTitle(
            "Take a snapshot of the mat from the overhead camera. Stop the live "
            "monitor first: a DirectShow camera opens once."
        )
        self.camera_id = QLineEdit(state.camera_instance_id)
        self.camera_id.setToolTip(
            "The camera's PnP instance id, e.g. USB\\VID_32E4&PID_5234&MI_00\\..."
        )
        self.camera_id.textChanged.connect(self._sync)
        self.width_px = self._spin(state.capture_width, MAX_CAPTURE_PX, "Width")
        self.height_px = self._spin(state.capture_height, MAX_CAPTURE_PX, "Height")
        self.fps_spin = self._spin(state.fps, MAX_FPS, "Rate")
        self.snapshot_button = QPushButton("Take snapshot")
        self.snapshot_button.setToolTip("Grab one settled frame from the camera.")
        self.snapshot_button.clicked.connect(self.take)
        self.load_button = QPushButton("Load image…")
        self.load_button.setToolTip("Use a saved frame instead of the camera.")
        self.load_button.clicked.connect(self._browse)
        self.status = QLabel("No snapshot yet.")
        self.status.setWordWrap(True)
        self.status.setToolTip("Snapshot size, or why it failed.")
        self.preview = CornerCanvas()
        self.preview.setEnabled(False)
        form = QFormLayout()
        form.setSpacing(TOKENS.spacing_px)
        form.addRow("Camera id", self.camera_id)
        form.addRow("Width (px)", self.width_px)
        form.addRow("Height (px)", self.height_px)
        form.addRow("Rate (fps)", self.fps_spin)
        buttons = QHBoxLayout()
        buttons.addWidget(self.snapshot_button)
        buttons.addWidget(self.load_button)
        buttons.addStretch()
        column = QVBoxLayout(self)
        column.setSpacing(TOKENS.spacing_px)
        column.addLayout(form)
        column.addLayout(buttons)
        column.addWidget(self.status)
        column.addWidget(self.preview, 1)
        for spin in (self.width_px, self.height_px, self.fps_spin):
            spin.valueChanged.connect(self._sync)

    @staticmethod
    def _spin(value: int, maximum: int, tip: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, maximum)
        spin.setValue(value)
        spin.setToolTip(f"{tip} the camera streams at; the calibration records it.")
        return spin

    def _sync(self) -> None:
        self.state.camera_instance_id = self.camera_id.text().strip()
        self.state.capture_width = self.width_px.value()
        self.state.capture_height = self.height_px.value()
        self.state.fps = self.fps_spin.value()
        self.completeChanged.emit()

    def mode(self) -> CaptureMode:
        return CaptureMode(
            self.width_px.value(), self.height_px.value(), self.fps_spin.value()
        )

    def take(self) -> None:
        self._sync()
        try:
            frame = self._snapshot(self.state.camera_instance_id, self.mode())
        except (OSError, StateError, ValueError, RuntimeError) as exc:
            logger.warning("snapshot failed: %s", exc)
            self.status.setText(f"Snapshot failed: {exc}")
            return
        self.set_frame(frame)

    def _browse(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Load a frame", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if name:
            self.load_file(Path(name))

    def load_file(self, path: Path) -> None:
        try:
            self.set_frame(load_image_file(path))
        except (OSError, StateError, ValueError) as exc:
            self.status.setText(f"Could not load {path.name}: {exc}")

    def set_frame(self, frame: Frame) -> None:
        self.state.frame = frame
        h, w = frame.shape[:2]
        self.preview.set_frame(frame)
        self.status.setText(f"Snapshot {w} x {h}.")
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802
        return bool(self.state.camera_instance_id) and self.state.frame is not None


class ReviewPage(QWizardPage):
    """Reprojection error and the search region, before saving."""

    def __init__(self, state: WizardState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.setTitle("Review")
        self.setSubTitle("Finish writes the calibration; the monitor reloads it.")
        self.canvas = CornerCanvas()
        self.canvas.setEnabled(False)
        self.error_px = QLabel()
        self.error_px.setToolTip(
            "RMS pixel error mapping the mat rectangle back onto the clicked "
            "corners. Above about 3 px, re-click the corners."
        )
        self.region = QLabel()
        self.region.setToolTip("Pixel rectangle the ball is searched in (dashed).")
        self.radius = QLabel()
        self.radius.setToolTip("Ball radius bounds in pixels, from the mat's scale.")
        form = QFormLayout()
        form.setSpacing(TOKENS.spacing_px)
        form.addRow("Reprojection error", self.error_px)
        form.addRow("Search region", self.region)
        form.addRow("Ball radius (px)", self.radius)
        column = QVBoxLayout(self)
        column.setSpacing(TOKENS.spacing_px)
        column.addLayout(form)
        column.addWidget(self.canvas, 1)

    def initializePage(self) -> None:  # noqa: N802
        cal = calibration_from_state(self.state)
        region = cal.search_region()
        detector = cal.detector()
        self.error_px.setText(f"{cal.reprojection_error_px():.3f} px")
        self.region.setText(
            f"x {region.x0}–{region.x1}, y {region.y0}–{region.y1}"
            + ("" if cal.roi else " (mat + 25 % margin)")
        )
        self.radius.setText(
            f"{detector.min_radius_px:.1f} – {detector.max_radius_px:.1f}"
        )
        self.canvas.set_frame(self.state.frame)
        self.canvas.corners = list(self.state.corners)
        self.canvas.set_region(region)


class CalibrationWizard(QWizard):
    """Snapshot -> corners -> mat -> review -> saved calibration."""

    def __init__(
        self,
        save_path: Path,
        *,
        initial: Calibration | None = None,
        snapshot: SnapshotFn = take_snapshot,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate the putting monitor")
        self.setWizardStyle(_STYLE)
        self.save_path = save_path
        self.state = WizardState()
        if initial is not None:
            self._seed(initial)
        self.camera_page = CameraPage(self.state, snapshot)
        self.corners_page = CornersPage(self.state)
        self.mat_page = MatPage(self.state)
        self.review_page = ReviewPage(self.state)
        for page in (
            self.camera_page,
            self.corners_page,
            self.mat_page,
            self.review_page,
        ):
            self.addPage(page)
        self.result_calibration: Calibration | None = None

    def _seed(self, cal: Calibration) -> None:
        s = self.state
        s.camera_instance_id = cal.camera_instance_id
        s.capture_width, s.capture_height, s.fps = (
            cal.capture_width,
            cal.capture_height,
            cal.fps,
        )
        s.mat_width_mm, s.mat_length_mm = cal.mat_width_mm, cal.mat_length_mm
        s.target_deg, s.colour, s.custom_colour = (
            cal.target_deg,
            cal.colour,
            cal.custom_colour,
        )
        s.notes = cal.notes
        if cal.roi is not None:
            s.roi = (cal.roi.x0, cal.roi.y0, cal.roi.x1, cal.roi.y1)

    def calibration(self) -> Calibration:
        """The calibration the current state describes (four corners needed)."""
        return calibration_from_state(self.state)

    def accept(self) -> None:
        """Postcondition: the calibration is on disk at ``save_path``."""
        cal = self.calibration()
        cal.save(self.save_path)
        self.result_calibration = cal
        logger.info("calibration saved to %s", self.save_path)
        super().accept()
