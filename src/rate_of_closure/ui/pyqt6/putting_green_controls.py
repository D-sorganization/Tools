"""Green-surface controls for the Putting tab (epic #4800, P6).

One group box owning the P2 surface and its capture model: green speed
(stimp), the planar grade and downhill aspect, the hole distance, the
hole-capture model, and the heightfield import seam.

The planar controls stay authoritative while the green is planar; an
imported heightfield replaces the geometry outright and the planar
grade/aspect are then disabled rather than silently ignored — the
displayed authority always names the surface actually integrated.

This module is binding only. Both readers behind
:func:`~rate_of_closure.putting.green_surface_from_document` are the
shared, versioned, fail-closed ones; nothing here parses geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QWidget,
)

from rate_of_closure.putting import green_surface_from_document
from shared.python.swing_sim.putting import (
    CaptureModel,
    GreenSurface,
    PlanarGreenSurface,
)

logger = logging.getLogger(__name__)

__all__ = ["PuttGreen", "PuttingGreenControls"]

#: Display label -> ``simulate_putt_on_surface`` capture model.
_CAPTURE_MODELS: tuple[tuple[str, CaptureModel], ...] = (
    ("Effective radius (Holmes/Penner)", "effective_radius"),
    ("Speed threshold (legacy #4125)", "speed_threshold"),
)

#: What a planar green declares as its provenance.
_PLANAR_SOURCE = "planar grade/aspect"


@dataclass(frozen=True)
class PuttGreen:
    """The green one putt is integrated on."""

    surface: GreenSurface
    source: str
    stimp_ft: float
    hole_distance_m: float
    capture_model: CaptureModel
    grade_percent: float
    aspect_deg: float

    def label(self) -> str:
        """Bounded scientific authority for the displayed result."""
        return (
            f"green {self.source}; stimp {self.stimp_ft:.2f} ft; grade "
            f"{self.grade_percent:.2f}%; aspect {self.aspect_deg:.1f} deg; "
            f"hole {self.hole_distance_m:.2f} m; capture "
            f"{self.capture_model}"
        )


class PuttingGreenControls(QGroupBox):
    """Green speed, slope, hole distance, capture model, and import."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Green And Hole", parent)
        self._imported: GreenSurface | None = None
        self._source = _PLANAR_SOURCE
        self._import_error = ""
        form = QFormLayout(self)
        self._build_surface_rows(form)
        self._build_capture_rows(form)
        self._build_import_rows(form)

    # ── construction ────────────────────────────────────────────────
    def _spin(
        self,
        low: float,
        high: float,
        value: float,
        step: float,
        suffix: str,
        tooltip: str,
        name: str,
        decimals: int = 2,
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setSuffix(suffix)
        box.setToolTip(tooltip)
        box.setAccessibleName(name)
        box.valueChanged.connect(self.changed)
        return box

    def _build_surface_rows(self, form: QFormLayout) -> None:
        self.stimp_spin = self._spin(
            3.0,
            16.0,
            10.0,
            0.5,
            " ft",
            "Green speed as a stimpmeter reading. Suggested range: "
            "7 (slow) - 13 (tournament fast). Source: USGA stimpmeter "
            "geometry (swing_sim.putting.roll derivation).",
            "Green Speed Stimp",
            decimals=1,
        )
        self.grade_spin = self._spin(
            0.0,
            10.0,
            0.0,
            0.25,
            " %",
            "Uniform slope grade of the planar green. Suggested range: "
            "0-4 % (greens rarely exceed ~5 %). Ignored while an "
            "imported heightfield is in use.",
            "Green Slope Grade",
        )
        self.aspect_spin = self._spin(
            -360.0,
            360.0,
            90.0,
            5.0,
            "°",
            "Downhill direction relative to the putt line: 0° = "
            "downhill straight ahead, +90° = low side on your left, "
            "180° = uphill putt. Source: swing_sim.putting.green frame.",
            "Green Downhill Direction",
            decimals=0,
        )
        self.distance_spin = self._spin(
            0.1,
            40.0,
            3.0,
            0.1,
            " m",
            "Distance from the ball to the hole centre along the "
            "target line. Suggested range: 1-15 m. Source: "
            "swing_sim.putting.green.",
            "Distance To Hole",
            decimals=1,
        )
        form.addRow("Green speed (stimp)", self.stimp_spin)
        form.addRow("Slope grade", self.grade_spin)
        form.addRow("Downhill direction", self.aspect_spin)
        form.addRow("Distance to hole", self.distance_spin)

    def _build_capture_rows(self, form: QFormLayout) -> None:
        self.capture_combo = QComboBox()
        for label, _model in _CAPTURE_MODELS:
            self.capture_combo.addItem(label)
        self.capture_combo.setToolTip(
            "How the hole swallows the ball. The published "
            "effective-radius model shrinks the usable 54 mm opening "
            "as the arrival speed rises (Holmes/Penner); the legacy "
            "#4125 model is a plain speed threshold at the mouth."
        )
        self.capture_combo.setAccessibleName("Hole Capture Model")
        self.capture_combo.currentIndexChanged.connect(self.changed)
        form.addRow("Hole capture", self.capture_combo)

    def _build_import_rows(self, form: QFormLayout) -> None:
        self.import_button = QPushButton("Import heightfield…")
        self.import_button.setToolTip(
            "Load a green heightfield: a swing_sim.green_surface/1 "
            "document, or an UpstreamDrift putting_green topography "
            "(#4800 P2/P9). The reader is chosen by the document's "
            "declared format and refuses anything it does not fully "
            "understand."
        )
        self.import_button.clicked.connect(self._choose_green_document)
        self.planar_button = QPushButton("Use planar green")
        self.planar_button.setToolTip(
            "Discard an imported heightfield and return to the planar "
            "grade/aspect green above."
        )
        self.planar_button.clicked.connect(self.use_planar_green)
        self.planar_button.setEnabled(False)
        self.source_label = QLabel(_PLANAR_SOURCE)
        self.source_label.setWordWrap(True)
        self.source_label.setToolTip(
            "The green geometry actually integrated. A heightfield "
            "loaded through swing_sim.green_surface/1 or an "
            "UpstreamDrift topography (#4800 P2/P9) replaces the "
            "planar grade and aspect."
        )
        form.addRow("Green surface", self.source_label)
        form.addRow(self.import_button, self.planar_button)

    # ── behaviour ───────────────────────────────────────────────────
    def capture_model(self) -> CaptureModel:
        """The selected capture model identifier."""
        return _CAPTURE_MODELS[self.capture_combo.currentIndex()][1]

    def surface(self) -> GreenSurface:
        """The imported heightfield, or the planar green from the spins."""
        if self._imported is not None:
            return self._imported
        return PlanarGreenSurface(
            grade_percent=float(self.grade_spin.value()),
            aspect_deg=float(self.aspect_spin.value()),
        )

    def green(self) -> PuttGreen:
        """Read the widgets into one frozen green description."""
        return PuttGreen(
            surface=self.surface(),
            source=self._source,
            stimp_ft=float(self.stimp_spin.value()),
            hole_distance_m=float(self.distance_spin.value()),
            capture_model=self.capture_model(),
            grade_percent=float(self.grade_spin.value()),
            aspect_deg=float(self.aspect_spin.value()),
        )

    def adopt_green_document(self, path: Path | str) -> str:
        """Import a green from disk through the declared-format reader.

        Args:
            path: A ``swing_sim.green_surface/1`` document or an
                UpstreamDrift ``_surface_io`` topography.

        Returns:
            The provenance label of the adopted surface.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the selected reader refuses the document.
            TypeError: If a document field has the wrong type.
        """
        text = Path(path).read_text(encoding="utf-8")
        surface, wire = green_surface_from_document(text)
        self._imported = surface
        self._source = f"{Path(path).name} via {wire}"
        self._import_error = ""
        self._publish_source()
        self.changed.emit()
        return self._source

    def use_planar_green(self) -> None:
        """Return to the planar grade/aspect green (idempotent)."""
        if self._imported is None:
            return
        self._imported = None
        self._source = _PLANAR_SOURCE
        self._import_error = ""
        self._publish_source()
        self.changed.emit()

    def import_error(self) -> str:
        """The last refused-import message, empty when none stands."""
        return self._import_error

    def _choose_green_document(self) -> None:
        """Open the file chooser and adopt the selection, or announce why not."""
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Import green heightfield",
            "",
            "Green surface documents (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.adopt_green_document(path)
        except (OSError, TypeError, ValueError) as error:
            logger.exception("green heightfield import refused")
            self._import_error = f"Green import refused ({Path(path).name}): {error}"
            self._publish_source()

    def _publish_source(self) -> None:
        imported = self._imported is not None
        self.source_label.setText(
            f"{self._source} — {self._import_error}"
            if self._import_error
            else self._source
        )
        self.planar_button.setEnabled(imported)
        for spin in (self.grade_spin, self.aspect_spin):
            spin.setEnabled(not imported)
