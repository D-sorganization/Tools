"""The calibration wizard's pages that gather input.

* :class:`MatDiagram` draws the corner-order convention: the mat as the
  player sees it, near edge at the bottom, corners numbered
  1 near-left, 2 near-right, 3 far-right, 4 far-left. That order is what
  lets a camera facing the player work with no flip flag.
* :class:`CornersPage` shows the snapshot and collects four clicks.
* :class:`MatPage` collects mat size, target angle, ball colour and an
  optional region of interest.

Pages read and write a shared :class:`WizardState`; the wizard turns the
state into a :class:`Calibration` in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QPainter, QPaintEvent, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from putting_launch_monitor.calibration import CORNER_ORDER
from putting_launch_monitor.detect import COLOUR_PROFILES, Frame, HsvRange
from shared.python.theme import Sizes, get_display_font

from .live_view import CornerCanvas
from .painting import TOKENS, OverlayPalette

_ANTIALIAS = QPainter.RenderHint.Antialiasing
_NO_BRUSH = Qt.BrushStyle.NoBrush
_ALIGN_TOP = Qt.AlignmentFlag.AlignTop

DEFAULT_MAT_MM = (1219.0, 1524.0)  # the lab's placeholder 4 x 5 ft mat
MAX_MAT_MM = 20_000.0
MAX_ROI_PX = 16_384


@dataclass
class WizardState:
    """Everything the wizard has gathered so far."""

    camera_instance_id: str = ""
    capture_width: int = 1920
    capture_height: int = 1200
    fps: int = 60
    frame: Frame | None = None
    corners: list[tuple[float, float]] = field(default_factory=list)
    mat_width_mm: float = DEFAULT_MAT_MM[0]
    mat_length_mm: float = DEFAULT_MAT_MM[1]
    target_deg: float = 0.0
    colour: str = "white"
    custom_colour: HsvRange | None = None
    roi: tuple[int, int, int, int] | None = None
    notes: str = ""


class MatDiagram(QWidget):
    """The corner-order convention, drawn."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.palette_ = OverlayPalette.from_theme()
        self.setMinimumSize(180, 200)
        self.setToolTip(
            "Corner order as the player sees it: 1 near-left, 2 near-right, "
            "3 far-right, 4 far-left. Near is the player's end, far is the "
            "target's. The camera may face any way; this order fixes the frame."
        )

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(_ANTIALIAS)
        w, h = self.width(), self.height()
        pad = TOKENS.padding_px * 3
        near_y, far_y = h - pad * 2, pad * 2
        near = (pad, w - pad)
        far = (pad + (w - 2 * pad) * 0.15, w - pad - (w - 2 * pad) * 0.15)
        corners = [
            QPointF(near[0], near_y),
            QPointF(near[1], near_y),
            QPointF(far[1], far_y),
            QPointF(far[0], far_y),
        ]
        pen = QPen(self.palette_.corner)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(_NO_BRUSH)
        painter.drawPolygon(QPolygonF(corners))
        painter.setFont(get_display_font(Sizes.SM))
        offsets = [(-12, 16), (0, 16), (0, -6), (-12, -6)]
        for i, (point, (dx, dy)) in enumerate(zip(corners, offsets, strict=True)):
            painter.drawEllipse(point, 5, 5)
            painter.drawText(
                QPointF(point.x() + dx, point.y() + dy), f"{i + 1} {CORNER_ORDER[i]}"
            )
        painter.setPen(QPen(self.palette_.text))
        painter.drawText(QPointF(pad, h - pad / 2), "player (near)")
        painter.drawText(QPointF(pad, pad), "target (far)")
        painter.end()


class CornersPage(QWizardPage):
    """Click the four corners on the snapshot, in order."""

    def __init__(self, state: WizardState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.setTitle("Mat corners")
        self.setSubTitle(
            "Click the four corners of the hitting mat in this order, as the "
            "player sees them: 1 near-left, 2 near-right, 3 far-right, 4 far-left."
        )
        self.canvas = CornerCanvas()
        self.canvas.corners_changed.connect(self._on_corners)
        self.diagram = MatDiagram()
        self.status = QLabel()
        self.status.setToolTip("How many corners have been clicked so far.")
        self.clear_button = QPushButton("Clear corners")
        self.clear_button.setToolTip("Discard the clicked corners and start again.")
        self.clear_button.clicked.connect(self.canvas.clear_corners)
        side = QVBoxLayout()
        side.setSpacing(TOKENS.spacing_px)
        side.addWidget(self.diagram)
        side.addWidget(self.status)
        side.addWidget(self.clear_button)
        side.addStretch()
        row = QHBoxLayout(self)
        row.setSpacing(TOKENS.padding_px)
        row.addWidget(self.canvas, 1)
        row.addLayout(side)
        self._on_corners()

    def initializePage(self) -> None:  # noqa: N802
        self.canvas.set_frame(self.state.frame)

    def _on_corners(self) -> None:
        self.state.corners = list(self.canvas.corners)
        n = len(self.state.corners)
        next_name = CORNER_ORDER[n] if n < 4 else "done"
        self.status.setText(f"{n} of 4 corners; next: {next_name}")
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802
        return len(self.state.corners) == 4


class MatPage(QWizardPage):
    """Mat size, target angle, ball colour, optional search region."""

    def __init__(self, state: WizardState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.setTitle("Mat and ball")
        self.setSubTitle(
            "Measure the mat with a tape: every speed scales with these numbers."
        )
        form = QFormLayout(self)
        form.setSpacing(TOKENS.spacing_px)
        self.width_mm = self._mm_spin(state.mat_width_mm)
        self.width_mm.setToolTip("Mat width in millimetres: near-left to near-right.")
        self.length_mm = self._mm_spin(state.mat_length_mm)
        self.length_mm.setToolTip("Mat length in millimetres: near edge to far edge.")
        self.target_deg = QDoubleSpinBox()
        self.target_deg.setRange(-89.0, 89.0)
        self.target_deg.setDecimals(1)
        self.target_deg.setValue(state.target_deg)
        self.target_deg.setToolTip(
            "Target line relative to the mat's long axis, degrees; positive "
            "toward the player's right. Zero for a hole straight down the mat."
        )
        self.colour = QComboBox()
        self.colour.addItems(sorted(COLOUR_PROFILES))
        self.colour.setCurrentText(state.colour)
        self.colour.setToolTip("Ball colour profile (HSV bounds); tune later.")
        self.notes = QLineEdit(state.notes)
        self.notes.setToolTip("Free text saved with the calibration.")
        self.roi_enabled = QCheckBox("Restrict the search to a region")
        self.roi_enabled.setToolTip(
            "Off: search the mat plus a 25 % margin. On: search only this "
            "pixel rectangle of the full-resolution frame."
        )
        self.roi = [self._px_spin(v) for v in (0, 0, state.capture_width, 1)]
        self.roi[3].setValue(state.capture_height)
        for spin, tip in zip(
            self.roi, ("left x", "top y", "right x", "bottom y"), strict=True
        ):
            spin.setToolTip(f"Region {tip}, in pixels of the full frame.")
        roi_grid = QGridLayout()
        roi_grid.setSpacing(TOKENS.spacing_px)
        names = ("x0", "y0", "x1", "y1")
        for i, (label, spin) in enumerate(zip(names, self.roi, strict=True)):
            roi_grid.addWidget(QLabel(label), 0, 2 * i)
            roi_grid.addWidget(spin, 0, 2 * i + 1)
        self.roi_enabled.toggled.connect(self._toggle_roi)
        self._toggle_roi(False)
        form.addRow("Mat width (mm)", self.width_mm)
        form.addRow("Mat length (mm)", self.length_mm)
        form.addRow("Target angle (deg)", self.target_deg)
        form.addRow("Ball colour", self.colour)
        form.addRow("Notes", self.notes)
        self.error = QLabel("")
        self.error.setToolTip("Why the page cannot be left yet.")
        form.addRow(self.roi_enabled)
        form.addRow(roi_grid)
        form.addRow(self.error)

    @staticmethod
    def _mm_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(1.0, MAX_MAT_MM)
        spin.setDecimals(1)
        spin.setValue(value)
        return spin

    @staticmethod
    def _px_spin(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, MAX_ROI_PX)
        spin.setValue(value)
        return spin

    def _toggle_roi(self, enabled: bool) -> None:
        for spin in self.roi:
            spin.setEnabled(enabled)

    def validatePage(self) -> bool:  # noqa: N802
        s = self.state
        s.mat_width_mm = self.width_mm.value()
        s.mat_length_mm = self.length_mm.value()
        s.target_deg = self.target_deg.value()
        s.colour = self.colour.currentText()
        s.notes = self.notes.text()
        if self.roi_enabled.isChecked():
            x0, y0, x1, y1 = (spin.value() for spin in self.roi)
            if x1 <= x0 or y1 <= y0:
                self.error.setText("The region must have positive width and height.")
                return False
            s.roi = (x0, y0, x1, y1)
        else:
            s.roi = None
        self.error.setText("")
        return True
