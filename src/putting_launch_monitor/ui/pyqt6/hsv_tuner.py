"""HSV tuning: six sliders bound to an :class:`HsvRange`, the mask beside the frame.

The mask is the detector's own (:meth:`HsvBallDetector.mask`) run on the
frame the operator supplied, so what is shown is exactly what the blob
search sees. Accepting hands back the range; the caller decides what to
do with it (the main window writes it into the calibration as
``custom_colour``).
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from putting_launch_monitor.detect import Frame, HsvBallDetector, HsvRange
from shared.python.contracts import require
from shared.python.theme import Sizes, get_mono_font

from .live_view import ImageCanvas
from .painting import TOKENS, mask_to_qimage

_HORIZONTAL = Qt.Orientation.Horizontal
_OK_CANCEL = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

# (field, label, maximum, tooltip)
BOUNDS: tuple[tuple[str, str, int, str], ...] = (
    ("h_min", "Hue min", 179, "Lowest hue kept (OpenCV 0..179; red wraps at 0)."),
    ("h_max", "Hue max", 179, "Highest hue kept."),
    ("s_min", "Sat min", 255, "Lowest saturation kept; a white ball sits near 0."),
    ("s_max", "Sat max", 255, "Highest saturation kept."),
    ("v_min", "Val min", 255, "Lowest brightness kept; raise it to drop the mat."),
    ("v_max", "Val max", 255, "Highest brightness kept."),
)


class HsvTuner(QWidget):
    """Sliders for an :class:`HsvRange`, with the resulting mask shown."""

    range_changed = pyqtSignal(object)

    def __init__(
        self,
        detector: HsvBallDetector,
        frame: Frame | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._detector = detector
        self._frame: Frame | None = None
        self.frame_view = ImageCanvas()
        self.frame_view.setToolTip("The frame being thresholded.")
        self.mask_view = ImageCanvas()
        self.mask_view.setToolTip(
            "The detector's binary mask: white where a pixel passes the HSV "
            "bounds after blur and morphology. The ball should be a clean disc."
        )
        self.sliders: dict[str, QSlider] = {}
        self.values: dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setSpacing(TOKENS.spacing_px)
        current = detector.colour
        for row, (field, label, maximum, tip) in enumerate(BOUNDS):
            slider = QSlider(_HORIZONTAL)
            slider.setRange(0, maximum)
            slider.setValue(getattr(current, field))
            slider.setToolTip(tip)
            slider.valueChanged.connect(self._on_slider)
            value = QLabel(str(slider.value()))
            value.setFont(get_mono_font(Sizes.BASE))
            value.setMinimumWidth(TOKENS.minimum_control_px // 2)
            self.sliders[field] = slider
            self.values[field] = value
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value, row, 2)
        views = QHBoxLayout()
        views.setSpacing(TOKENS.spacing_px)
        views.addWidget(self.frame_view, 1)
        views.addWidget(self.mask_view, 1)
        column = QVBoxLayout(self)
        column.setSpacing(TOKENS.spacing_px)
        column.addLayout(views, 1)
        column.addLayout(grid)
        self.set_frame(frame)

    # -- state ----------------------------------------------------------------------
    def hsv_range(self) -> HsvRange:
        """The range the sliders describe (min/max swapped into order if crossed)."""
        v = {field: slider.value() for field, slider in self.sliders.items()}
        for lo, hi in (("h_min", "h_max"), ("s_min", "s_max"), ("v_min", "v_max")):
            if v[lo] > v[hi]:
                v[lo], v[hi] = v[hi], v[lo]
        return HsvRange(**v)

    def set_range(self, colour: HsvRange) -> None:
        for field, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setValue(getattr(colour, field))
            slider.blockSignals(False)
        self._on_slider()

    def detector(self) -> HsvBallDetector:
        """The bound detector with the sliders' colour."""
        return replace(self._detector, colour=self.hsv_range())

    def set_frame(self, frame: Frame | None) -> None:
        if frame is not None:
            require(frame.ndim == 3 and frame.shape[2] == 3, "BGR frame", frame.shape)
        self._frame = frame
        self.frame_view.set_frame(frame)
        self.refresh_mask()

    def refresh_mask(self) -> None:
        if self._frame is None:
            self.mask_view.set_image(None)
            return
        self.mask_view.set_image(mask_to_qimage(self.detector().mask(self._frame)))

    def _on_slider(self) -> None:
        for field, slider in self.sliders.items():
            self.values[field].setText(str(slider.value()))
        self.refresh_mask()
        self.range_changed.emit(self.hsv_range())


class HsvTunerDialog(QDialog):
    """A modal wrapper: OK returns the tuned range through :meth:`hsv_range`."""

    def __init__(
        self,
        detector: HsvBallDetector,
        frame: Frame | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tune ball colour")
        self.tuner = HsvTuner(detector, frame)
        buttons = QDialogButtonBox(_OK_CANCEL)
        buttons.setToolTip("OK writes the bounds into the calibration.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column = QVBoxLayout(self)
        column.setSpacing(TOKENS.spacing_px)
        column.addWidget(self.tuner, 1)
        column.addWidget(buttons)

    def hsv_range(self) -> HsvRange:
        return self.tuner.hsv_range()
