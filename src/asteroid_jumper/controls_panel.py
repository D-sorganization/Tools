"""Controls panel — sliders and buttons for the simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from asteroid_jumper.controller import SimController

from asteroid_jumper.asteroid_shape import ShapeKind
from asteroid_jumper.controller import (
    DEFAULT_ASTEROID_RADIUS,
    DEFAULT_IMPULSE,
    JUMPER_MASS,
)


class ControlsPanel(QWidget):
    """Left-side controls: shape, mass, jump parameters, buttons."""

    jump_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    config_changed = pyqtSignal()

    def __init__(
        self, controller: SimController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        if not (controller is not None):
            raise ValueError("DbC Blocked: Precondition failed.")
        self._ctrl = controller
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_force_angle(self, angle_deg: float) -> None:
        """Update force-angle display when mouse drag triggers change."""
        if not (angle_deg is not None):
            raise ValueError("angle_deg must be provided")
        self._force_angle_spin.blockSignals(True)
        self._force_angle_spin.setValue(angle_deg)
        self._force_angle_spin.blockSignals(False)

    def sync_from_controller(self) -> None:
        """Refresh all controls from current controller state."""
        self._asteroid_mass_spin.setValue(self._ctrl.asteroid_mass)
        self._impulse_spin.setValue(self._ctrl.impulse_magnitude)
        self._force_angle_spin.setValue(self._ctrl.force_angle_deg)

    def enable_controls(self, enabled: bool) -> None:
        """Lock controls during flight."""
        if not (enabled is not None):
            raise ValueError("enabled must be provided")
        self._asteroid_mass_spin.setEnabled(enabled)
        self._shape_combo.setEnabled(enabled)
        self._semi_a_spin.setEnabled(enabled)
        self._semi_b_spin.setEnabled(enabled)
        self._impulse_spin.setEnabled(enabled)
        self._force_angle_spin.setEnabled(enabled)
        self._jump_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Private — UI build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(self._build_asteroid_group())
        layout.addWidget(self._build_jump_group())
        layout.addWidget(self._build_buttons_group())
        layout.addStretch()

    def _build_asteroid_group(self) -> QGroupBox:
        group = QGroupBox("Asteroid")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        # Shape selector
        vbox.addWidget(QLabel("Shape:"))
        self._shape_combo = QComboBox()
        self._shape_combo.addItems(["Ellipse", "Circle", "Random"])
        self._shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        vbox.addWidget(self._shape_combo)

        # Semi-a
        vbox.addWidget(QLabel("Semi-axis A (m):"))
        self._semi_a_spin = _make_dspin(1.0, 50.0, DEFAULT_ASTEROID_RADIUS, 0.5)
        self._semi_a_spin.valueChanged.connect(self._on_shape_param_changed)
        vbox.addWidget(self._semi_a_spin)

        # Semi-b
        vbox.addWidget(QLabel("Semi-axis B (m):"))
        self._semi_b_spin = _make_dspin(1.0, 50.0, DEFAULT_ASTEROID_RADIUS * 0.6, 0.5)
        self._semi_b_spin.valueChanged.connect(self._on_shape_param_changed)
        vbox.addWidget(self._semi_b_spin)

        # Mass
        vbox.addWidget(QLabel(f"Asteroid mass (kg)  [jumper = {JUMPER_MASS:.0f} kg]:"))
        self._asteroid_mass_spin = _make_dspin(
            20.0, 5000.0, JUMPER_MASS * 2.0, 10.0, decimals=0
        )
        self._asteroid_mass_spin.valueChanged.connect(self._on_mass_changed)
        vbox.addWidget(self._asteroid_mass_spin)

        return group

    def _build_jump_group(self) -> QGroupBox:
        group = QGroupBox("Jump Parameters")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        # Force angle (also set by mouse drag)
        vbox.addWidget(QLabel("Force angle (°): [drag on asteroid to set]"))
        self._force_angle_spin = _make_dspin(-180.0, 180.0, 90.0, 1.0)
        self._force_angle_spin.valueChanged.connect(self._on_force_angle_changed)
        vbox.addWidget(self._force_angle_spin)

        # Impulse slider
        vbox.addWidget(QLabel("Jump impulse (N·s):"))
        self._impulse_spin = _make_dspin(
            50.0, 2000.0, DEFAULT_IMPULSE, 10.0, decimals=0
        )
        self._impulse_spin.valueChanged.connect(self._on_impulse_changed)
        vbox.addWidget(self._impulse_spin)

        # Off-centre hint
        self._offcentre_label = QLabel("Off-centre: 0%")
        self._offcentre_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self._offcentre_label)

        return group

    def _build_buttons_group(self) -> QGroupBox:
        group = QGroupBox("Simulation")
        vbox = QVBoxLayout(group)

        self._jump_btn = QPushButton("🚀  JUMP!")
        self._jump_btn.setFixedHeight(40)
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        self._jump_btn.setFont(font)
        self._jump_btn.clicked.connect(self.jump_requested)
        vbox.addWidget(self._jump_btn)

        self._reset_btn = QPushButton("↺  Reset")
        self._reset_btn.clicked.connect(self.reset_requested)
        vbox.addWidget(self._reset_btn)

        return group

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_shape_changed(self, index: int) -> None:
        if not (index is not None):
            raise ValueError("index must be provided")
        shapes = [ShapeKind.ELLIPSE, ShapeKind.CIRCLE, ShapeKind.RANDOM]
        self._ctrl.configure(shape_kind=shapes[index])
        self._update_offcentre()
        self.config_changed.emit()

    def _on_shape_param_changed(self) -> None:
        self._ctrl.configure(
            semi_a=self._semi_a_spin.value(),
            semi_b=self._semi_b_spin.value(),
        )
        self._update_offcentre()
        self.config_changed.emit()

    def _on_mass_changed(self, value: float) -> None:
        self._ctrl.configure(asteroid_mass=value)
        self.config_changed.emit()

    def _on_force_angle_changed(self, value: float) -> None:
        if not (value is not None):
            raise ValueError("value must be provided")
        self._ctrl.set_force_angle(value)
        self._ctrl.set_jump_direction(value)
        self._ctrl.state = self._ctrl._build_state()
        self._update_offcentre()
        self.config_changed.emit()

    def _on_impulse_changed(self, value: float) -> None:
        self._ctrl.set_impulse(value)
        self.config_changed.emit()

    def _update_offcentre(self) -> None:
        frac = self._ctrl.off_centre_fraction()
        self._offcentre_label.setText(f"Off-centre: {frac:.1%}")


def _make_dspin(
    lo: float,
    hi: float,
    default: float,
    step: float,
    decimals: int = 2,
) -> QDoubleSpinBox:
    """Create a configured QDoubleSpinBox."""
    if not (lo is not None):
        raise ValueError("lo must be provided")
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(default)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    return spin
