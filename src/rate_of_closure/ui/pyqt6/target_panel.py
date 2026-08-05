"""Target-region editor + 'Optimize to Target' (epic #4125, H7b).

A compact group box the Solver panel embeds: choose the region kind
(green circle at a distance / fairway corridor), edit its geometry with
typed entries (the cheap place/edit seam — the flight top-down view
renders the region live as the entries change), weight it, and launch
the solver with the region goal wired into the existing partition /
progress machinery via :attr:`optimizeRequested`.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QWidget,
)

from rate_of_closure.simulation.targets import TargetRegion
from rate_of_closure.units import DISTANCE_UNITS, display_distance_unit

logger = logging.getLogger(__name__)

__all__ = ["TargetPanel"]

_KIND_GUIDANCE = (
    "Suggested range: Green for approach practice (circle at a distance "
    "with a radius), Fairway for tee shots (a carry-distance band with a "
    "half-width about the target line). Source: standard course-target "
    "geometry; solver region-goal documentation (#4125 H7b)."
)
_DISTANCE_GUIDANCE = (
    "Suggested range: 100-280 m downrange center of the target — green "
    "center distance or fairway band midpoint. Source: typical par-3 "
    "through driver landing zones."
)
_RADIUS_GUIDANCE = (
    "Suggested range: 5-15 m green radius (10 m default — a generous "
    "tour green). Source: published green-size surveys."
)
_LATERAL_GUIDANCE = (
    "Suggested range: within +/-30 m; positive moves the green right of "
    "the target line. Source: solver landing-plane sign convention."
)
_BAND_GUIDANCE = (
    "Suggested range: 10-30 m half-length of the fairway distance band "
    "(how much short/long is acceptable). Source: typical landing-zone "
    "depth guidance."
)
_WIDTH_GUIDANCE = (
    "Suggested range: 10-25 m fairway half-width (16 m default — a "
    "~35 yd fairway). Source: published course-architecture widths."
)
_WEIGHT_GUIDANCE = (
    "Suggested range: 0.5-5 weight of the region residual against any "
    "checked quantity goals (residual = distance outside the region, 0 "
    "inside, plus a small centering pull). Source: solver region-goal "
    "documentation (#4125 H7b)."
)


def make_spin(
    lo: float, hi: float, value: float, decimals: int, suffix: str
) -> QDoubleSpinBox:
    """A no-arrow, typed QDoubleSpinBox in the app's input style."""
    spin = QDoubleSpinBox()
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setKeyboardTracking(False)
    spin.setDecimals(decimals)
    spin.setRange(lo, hi)
    spin.setSuffix(suffix)
    spin.setValue(value)
    spin.setMinimumWidth(84)  # readable at small windows (#4120)
    return spin


class TargetPanel(QGroupBox):
    """Region kind + geometry entries + weight + 'Optimize to Target'."""

    #: Emitted with the freshly built TargetRegion after any edit.
    regionChanged = pyqtSignal(object)  # noqa: N815 — Qt convention
    #: Emitted when 'Optimize to Target' is clicked.
    optimizeRequested = pyqtSignal()  # noqa: N815 — Qt convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Target Region (Optimize to Target)", parent)
        grid = QGridLayout(self)
        grid.setVerticalSpacing(4)

        self._kind = QComboBox()
        self._kind.addItems(["Green (circle)", "Fairway (corridor)"])
        self._kind.setToolTip(_KIND_GUIDANCE)
        grid.addWidget(QLabel("Kind"), 0, 0)
        grid.addWidget(self._kind, 0, 1)

        self._distance = make_spin(20.0, 400.0, 230.0, 1, " m")
        self._distance.setToolTip(_DISTANCE_GUIDANCE)
        grid.addWidget(QLabel("Distance"), 1, 0)
        grid.addWidget(self._distance, 1, 1)

        self._radius = make_spin(1.0, 50.0, 10.0, 1, " m")
        self._radius.setToolTip(_RADIUS_GUIDANCE)
        self._radius_label = QLabel("Radius")
        grid.addWidget(self._radius_label, 2, 0)
        grid.addWidget(self._radius, 2, 1)

        self._lateral = make_spin(-60.0, 60.0, 0.0, 1, " m")
        self._lateral.setToolTip(_LATERAL_GUIDANCE)
        self._lateral_label = QLabel("Lateral Offset")
        grid.addWidget(self._lateral_label, 3, 0)
        grid.addWidget(self._lateral, 3, 1)

        self._band = make_spin(2.0, 80.0, 15.0, 1, " m")
        self._band.setToolTip(_BAND_GUIDANCE)
        self._band_label = QLabel("Band Half-Length")
        grid.addWidget(self._band_label, 4, 0)
        grid.addWidget(self._band, 4, 1)

        self._width = make_spin(2.0, 60.0, 16.0, 1, " m")
        self._width.setToolTip(_WIDTH_GUIDANCE)
        self._width_label = QLabel("Half-Width")
        grid.addWidget(self._width_label, 5, 0)
        grid.addWidget(self._width, 5, 1)

        self._weight = make_spin(0.01, 100.0, 1.0, 2, "")
        self._weight.setToolTip(_WEIGHT_GUIDANCE)
        grid.addWidget(QLabel("Weight"), 6, 0)
        grid.addWidget(self._weight, 6, 1)

        self._optimize = QPushButton("Optimize to Target")
        self._optimize.setToolTip(
            "Run the solver with the target-region goal added to any "
            "checked quantity goals: the residual is the landing point's "
            "distance outside the region (0 inside, small centering "
            "pull), reusing the partition, multi-start, progress, and "
            "cancel machinery."
        )
        self._optimize.clicked.connect(self.optimizeRequested)
        grid.addWidget(self._optimize, 7, 0, 1, 2)

        self._kind.currentIndexChanged.connect(self._sync_kind)
        for spin in (
            self._distance,
            self._radius,
            self._lateral,
            self._band,
            self._width,
            self._weight,
        ):
            spin.valueChanged.connect(self._emit_region)
        self._kind.currentIndexChanged.connect(self._emit_region)
        self._sync_kind()
        # Distance display unit (#4125 H6): entries follow the session
        # unit (yards default); region() always reports canonical metres.
        self._distance_unit = "m"
        self.refresh_units()

    # ── public API ──────────────────────────────────────────────────
    def region(self) -> TargetRegion:
        """The TargetRegion described by the entries (DbC-validated).

        Entries display in the session distance unit; the region is
        always canonical SI metres.
        """
        factor = DISTANCE_UNITS[self._distance_unit]
        if self._kind.currentIndex() == 0:
            return TargetRegion(
                kind="green",
                distance_m=self._distance.value() * factor,
                radius_m=self._radius.value() * factor,
                lateral_m=self._lateral.value() * factor,
            )
        return TargetRegion(
            kind="fairway",
            distance_m=self._distance.value() * factor,
            band_half_length_m=self._band.value() * factor,
            half_width_m=self._width.value() * factor,
        )

    def refresh_units(self) -> None:
        """Re-display the geometry entries in the session distance unit."""
        unit = display_distance_unit()
        if unit == self._distance_unit:
            return
        old = DISTANCE_UNITS[self._distance_unit]
        new = DISTANCE_UNITS[unit]
        spins = (self._distance, self._radius, self._lateral, self._band, self._width)
        for spin in spins:
            spin.blockSignals(True)
            canonical = spin.value() * old
            low, high = spin.minimum() * old, spin.maximum() * old
            spin.setRange(low / new, high / new)
            spin.setValue(canonical / new)
            spin.setSuffix(f" {unit}")
            spin.blockSignals(False)
        self._distance_unit = unit
        self._emit_region()

    def weight(self) -> float:
        """The region-goal weight."""
        return float(self._weight.value())

    def optimize_button(self) -> QPushButton:
        """The 'Optimize to Target' button (test seam)."""
        return self._optimize

    def set_running(self, running: bool) -> None:
        """Disable the optimize entry point while a solve is running."""
        self._optimize.setEnabled(not running)

    # ── internals ──────────────────────────────────────────────────
    def _sync_kind(self, *_args: object) -> None:
        green = self._kind.currentIndex() == 0
        for widget in (
            self._radius,
            self._radius_label,
            self._lateral,
            self._lateral_label,
        ):
            widget.setVisible(green)
        for widget in (self._band, self._band_label, self._width, self._width_label):
            widget.setVisible(not green)

    def _emit_region(self, *_args: object) -> None:
        self.regionChanged.emit(self.region())
