"""Delivered-stroke controls for the Putting tab (epic #4800, P6).

One group box owning everything the P1 impact solve takes: the putter
head (P3 ``PutterHeadDocument`` — library fallback, or a mesh-derived
head), the stroke pace (head speed, or a backstroke length through the
pendulum proxy), the delivery angles (shaft lean, aim, face, path,
attack) and the strike location on the face.

Sign conventions are P1's verbatim, which are ``swing_sim.impact``'s
verbatim (AffineDrift frame): aim/face/path positive to the **right**
of the target line, attack positive **up**, strike offsets positive
toward the **toe** and **high** on the face. Every spin box's range is
the matching :func:`~shared.python.swing_sim.putting.strike` bound, so
the widget cannot express a stroke the physics refuses.

This module is binding only: it reads widget state into one frozen
:class:`PuttStroke` record and forwards it. No physics is computed
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.putting import putter_head_documents
from shared.python.swing_sim.putting import clubhead_speed_from_backstroke

if TYPE_CHECKING:  # pragma: no cover - typing-only; the head import is lazy
    from shared.python.golf_club.putter_head import PutterHeadDocument

__all__ = ["PuttStroke", "PuttingStrokeControls"]

#: Standard 35 in putter as a pendulum length [m] (P1's own default).
_PUTTER_LENGTH_M = 0.889


@dataclass(frozen=True)
class PuttStroke:
    """One delivered putting stroke, in P1's units and sign conventions."""

    putter_name: str
    clubhead_speed_mps: float
    shaft_lean_deg: float
    aim_deg: float
    face_angle_deg: float
    path_angle_deg: float
    attack_angle_deg: float
    strike_offset_toe_mm: float
    strike_offset_high_mm: float

    def label(self) -> str:
        """Bounded scientific authority for the displayed result."""
        return (
            f"speed {self.clubhead_speed_mps:.3f} m/s; lean "
            f"{self.shaft_lean_deg:.1f} deg; aim {self.aim_deg:.1f} deg; "
            f"face {self.face_angle_deg:.1f} deg; path "
            f"{self.path_angle_deg:.1f} deg; attack "
            f"{self.attack_angle_deg:.1f} deg; strike "
            f"({self.strike_offset_toe_mm:.1f}, "
            f"{self.strike_offset_high_mm:.1f}) mm toe/high"
        )


class PuttingStrokeControls(QGroupBox):
    """Putter, pace, delivery angles, and strike location."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Putter And Stroke", parent)
        self._heads: dict[str, PutterHeadDocument] = putter_head_documents()
        form = QFormLayout(self)
        self._build_putter_row(form)
        self._build_pace_rows(form)
        self._build_delivery_rows(form)
        self._build_strike_rows(form)

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

    def _build_putter_row(self, form: QFormLayout) -> None:
        self.putter_combo = QComboBox()
        self.putter_combo.addItems(list(self._heads))
        self.putter_combo.setToolTip(
            "Putter head driving the impact solve. Library heads (H1 "
            "club library) carry no inertia tensor and use the "
            "documented catalogue MOI; a head imported from an STL "
            "carries its own mesh-derived tensor (golf_club."
            "putter_head/1, #4800 P3)."
        )
        self.putter_combo.setAccessibleName("Putter Head")
        self.putter_combo.currentTextChanged.connect(self.changed)
        form.addRow("Putter", self.putter_combo)

    def _build_pace_rows(self, form: QFormLayout) -> None:
        self.pace_mode = QComboBox()
        self.pace_mode.addItems(["Clubhead speed", "Backstroke length"])
        self.pace_mode.setToolTip(
            "How the stroke pace is set: the head speed at impact "
            "directly, or a pendulum backstroke length through "
            "v = A·sqrt(g/L) (simple-pendulum proxy)."
        )
        self.pace_mode.setAccessibleName("Putter Pace Input Mode")
        self.speed_spin = self._spin(
            0.2,
            6.0,
            1.8,
            0.05,
            " m/s",
            "Clubhead speed at impact. Suggested range: 0.5-3 m/s for "
            "putts inside 15 m. Source: pendulum-stroke kinematics "
            "(swing_sim.putting.impact).",
            "Putter Clubhead Speed",
        )
        self.backstroke_spin = self._spin(
            5.0,
            100.0,
            30.0,
            1.0,
            " cm",
            "Backstroke arc length; converted to head speed with the "
            "simple-pendulum proxy v = A·sqrt(g/L). Suggested range: "
            "10-60 cm. Source: swing_sim.putting.impact derivation.",
            "Putter Backstroke Length",
            decimals=0,
        )
        self.pace_stack = QStackedWidget()
        self.pace_stack.setToolTip(
            "Stroke pace entry — switches with the pace-input mode."
        )
        for widget in (self.speed_spin, self.backstroke_spin):
            holder = QWidget()
            holder_layout = QVBoxLayout(holder)
            holder_layout.setContentsMargins(0, 0, 0, 0)
            holder_layout.addWidget(widget)
            self.pace_stack.addWidget(holder)
        self.pace_stack.setCurrentIndex(0)
        self.pace_mode.currentIndexChanged.connect(self.pace_stack.setCurrentIndex)
        self.pace_mode.currentIndexChanged.connect(self.changed)
        form.addRow("Pace input", self.pace_mode)
        form.addRow("Stroke pace", self.pace_stack)

    def _build_delivery_rows(self, form: QFormLayout) -> None:
        self.lean_spin = self._spin(
            -10.0,
            10.0,
            0.0,
            0.5,
            "°",
            "Forward press at impact; negative de-lofts the face. "
            "Effective loft is the putter's static loft plus this "
            "lean (swing_sim.putting.impact).",
            "Putter Shaft Lean",
            decimals=1,
        )
        self.aim_spin = self._spin(
            -45.0,
            45.0,
            0.0,
            0.5,
            "°",
            "Where the whole stroke is aimed relative to the target "
            "line; + = right. Face and path are measured off this aim "
            "line, so a square stroke aimed 2° left starts exactly 2° "
            "left (swing_sim.putting.impact, #4800 P1).",
            "Putt Aim Angle",
            decimals=1,
        )
        self.face_spin = self._spin(
            -20.0,
            20.0,
            0.0,
            0.25,
            "°",
            "Face angle at impact relative to the aim line; + = open "
            "(right). The normal impulse launches the ball along the "
            "face, so face angle dominates the start line.",
            "Putter Face Angle",
            decimals=2,
        )
        self.path_spin = self._spin(
            -20.0,
            20.0,
            0.0,
            0.25,
            "°",
            "Putter path at impact relative to the aim line; + = "
            "in-to-out. The 2/7 tangential impulse pulls the start "
            "line a little toward the path and leaves the sidespin.",
            "Putter Path Angle",
            decimals=2,
        )
        self.attack_spin = self._spin(
            -10.0,
            10.0,
            0.0,
            0.25,
            "°",
            "Attack angle at impact; + = hitting up. Spin loft is the "
            "effective loft minus this angle, so hitting up trims the "
            "backspin the skid phase starts from.",
            "Putter Attack Angle",
            decimals=2,
        )
        for label, widget in (
            ("Shaft lean", self.lean_spin),
            ("Aim (+ right)", self.aim_spin),
            ("Face angle (+ open)", self.face_spin),
            ("Path (+ in-to-out)", self.path_spin),
            ("Attack angle (+ up)", self.attack_spin),
        ):
            form.addRow(label, widget)

    def _build_strike_rows(self, form: QFormLayout) -> None:
        self.toe_spin = self._spin(
            -40.0,
            40.0,
            0.0,
            1.0,
            " mm",
            "Strike location across the face; + = toward the toe. "
            "Off-centre strikes cut the head's effective mass along "
            "the contact normal, so the ball leaves slower.",
            "Putter Strike Toe Offset",
            decimals=1,
        )
        self.high_spin = self._spin(
            -20.0,
            20.0,
            0.0,
            1.0,
            " mm",
            "Strike location up the face; + = high. High strikes "
            "twist the head about the heel-toe axis, adding dynamic "
            "loft (golf_club.putter_head twist model, #4800 P3).",
            "Putter Strike High Offset",
            decimals=1,
        )
        form.addRow("Strike toe offset", self.toe_spin)
        form.addRow("Strike high offset", self.high_spin)

    # ── behaviour ───────────────────────────────────────────────────
    def clubhead_speed_mps(self) -> float:
        """Resolved head speed [m/s] for the selected pace mode."""
        if self.pace_mode.currentIndex() == 1:
            return float(
                clubhead_speed_from_backstroke(
                    self.backstroke_spin.value() / 100.0, _PUTTER_LENGTH_M
                )
            )
        return float(self.speed_spin.value())

    def stroke(self) -> PuttStroke:
        """Read the widgets into one frozen delivered stroke."""
        return PuttStroke(
            putter_name=self.putter_combo.currentText(),
            clubhead_speed_mps=self.clubhead_speed_mps(),
            shaft_lean_deg=float(self.lean_spin.value()),
            aim_deg=float(self.aim_spin.value()),
            face_angle_deg=float(self.face_spin.value()),
            path_angle_deg=float(self.path_spin.value()),
            attack_angle_deg=float(self.attack_spin.value()),
            strike_offset_toe_mm=float(self.toe_spin.value()),
            strike_offset_high_mm=float(self.high_spin.value()),
        )

    def head_document(self) -> PutterHeadDocument:
        """The selected ``PutterHeadDocument`` (LoD seam for the tab)."""
        return self._heads[self.putter_combo.currentText()]

    def adopt_putter_mesh(
        self,
        stl_path: Path | str,
        *,
        loft_deg: float,
        target_mass_kg: float,
    ) -> str:
        """Import an STL head through P3 and select it.

        The C1 mesh authority owns the volume/CG/tensor solve and the
        SHA-256 provenance; this method only names the entry and picks
        it. The display name is the file stem, so re-importing the same
        file replaces rather than duplicates the entry.

        Args:
            stl_path: Binary STL authored in the documented head frame.
            loft_deg: Static face loft of the imported head [deg].
            target_mass_kg: Scale selector — the head's known mass.

        Returns:
            The display name of the adopted head.

        Raises:
            ValueError: If the mesh or the scale selector is refused.
            OSError: If the file cannot be read.
        """
        from shared.python.golf_club.putter_head import putter_head_from_stl

        path = Path(stl_path)
        document = putter_head_from_stl(
            path.stem,
            path,
            loft_deg=loft_deg,
            target_mass_kg=target_mass_kg,
        )
        name = str(document.name)
        self._heads[name] = document
        if self.putter_combo.findText(name) < 0:
            self.putter_combo.addItem(name)
        self.putter_combo.setCurrentText(name)
        return name
