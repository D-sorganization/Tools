"""Accessible PyQt editor for the canonical physical ball setup."""

from __future__ import annotations

import sys

from PyQt6.QtCore import QEvent, QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QFocusEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QWidget,
)

from shared.python.swing_sim.ball_setup import BallSetup, BallSupportMode

__all__ = ["BallSetupControl"]

_TEE_HEIGHT_GUIDANCE = (
    "Suggested range: 0–100 mm. Tee Height is measured vertically from "
    "the ground plane to the bottom of the ball. Source: Tools #4143 "
    "physical ball-support convention."
)


class _WholeFieldSpinBox(QDoubleSpinBox):
    """Select the complete numeric field whenever it receives user focus."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.installEventFilter(self)

    def focusInEvent(self, event: QFocusEvent | None) -> None:  # noqa: N802
        super().focusInEvent(event)
        QTimer.singleShot(0, self._select_number)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        super().mousePressEvent(event)
        QTimer.singleShot(0, self._select_number)

    def eventFilter(  # noqa: N802
        self, watched: QObject | None, event: QEvent | None
    ) -> bool:
        line_edit = self.lineEdit()
        if (
            watched is line_edit
            and event is not None
            and event.type() is QEvent.Type.MouseButtonPress
        ):
            QTimer.singleShot(0, self._select_number)
        return super().eventFilter(watched, event)

    def _select_number(self) -> None:
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.selectAll()


class BallSetupControl(QGroupBox):
    """Edit Ground/Tee support while tracking club defaults explicitly."""

    setupChanged = pyqtSignal(object)  # noqa: N815 - Qt signal convention

    def __init__(
        self,
        club_default: BallSetup,
        club_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Ball Setup", parent)
        self._club_default = club_default
        self._club_name = club_name
        self._updating = False
        self._last_tee_height_mm = max(club_default.tee_height_m * 1000.0, 38.1)

        self._use_default = QCheckBox("Use Club Default")
        self._use_default.setChecked(True)
        self._use_default.setAccessibleName("Use Club Ball Setup Default")
        self._use_default.setToolTip(
            "Apply the selected club's representative Ground or Tee setup; "
            "editing either setup field creates an explicit override. "
            "Source: Tools #4143 club-derived defaults."
        )

        self._mode = QComboBox()
        self._mode.addItem("Ground", BallSupportMode.GROUND)
        self._mode.addItem("Tee", BallSupportMode.TEE)
        self._mode.setAccessibleName("Ball Support Mode")
        self._mode.setAccessibleDescription(
            "Choose whether the ball rests on the ground or on a physical tee."
        )
        self._mode.setToolTip(
            "Choose Ground or Tee support. Ground places the bottom of the "
            "ball on the ground plane. Source: Tools #4143 physical setup contract."
        )

        self._tee_height = _WholeFieldSpinBox()
        self._tee_height.setDecimals(2)
        self._tee_height.setRange(0.0, sys.float_info.max)
        self._tee_height.setSingleStep(0.1)
        self._tee_height.setSuffix(" mm")
        self._tee_height.setKeyboardTracking(False)
        self._tee_height.setAccessibleName("Tee Height in Millimetres")
        self._tee_height.setAccessibleDescription(
            "Vertical clearance from the ground plane to the bottom of the ball."
        )

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setAccessibleName("Ball Setup Status")

        form = QFormLayout(self)
        form.addRow(self._use_default)
        form.addRow("Support Mode", self._mode)
        form.addRow("Tee Height", self._tee_height)
        form.addRow(self._status)
        QWidget.setTabOrder(self._use_default, self._mode)
        QWidget.setTabOrder(self._mode, self._tee_height)

        self._use_default.toggled.connect(self._on_default_toggled)
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        self._tee_height.valueChanged.connect(self._on_height_changed)
        self._apply_setup(club_default)

    def setup(self) -> BallSetup:
        """Return the canonical validated setup represented by the controls."""
        mode = BallSupportMode(self._mode.currentData())
        height_m = self._tee_height.value() / 1000.0
        if mode is BallSupportMode.GROUND:
            height_m = 0.0
        return BallSetup(mode, height_m)

    def apply_club_default(self, setup: BallSetup, club_name: str) -> None:
        """Update the club-derived default, preserving any explicit override."""
        self._club_default = setup
        self._club_name = club_name
        if setup.support_mode is BallSupportMode.TEE:
            self._last_tee_height_mm = setup.tee_height_m * 1000.0
        if self._use_default.isChecked():
            self._apply_setup(setup)
        else:
            self._update_enabled_and_status()

    def set_setup(self, setup: BallSetup) -> None:
        """Load a canonical persisted setup as an explicit user override."""
        self._updating = True
        try:
            self._use_default.setChecked(False)
        finally:
            self._updating = False
        self._apply_setup(setup)

    def mode_combo(self) -> QComboBox:
        """Return the support-mode editor for integration tests and hosts."""
        return self._mode

    def tee_height_spin(self) -> QDoubleSpinBox:
        """Return the unit-labelled tee-height editor."""
        return self._tee_height

    def use_club_default_check(self) -> QCheckBox:
        """Return the explicit default/override toggle."""
        return self._use_default

    def interactive_widgets(self) -> tuple[QWidget, ...]:
        """Return every keyboard-interactive control in tab order."""
        return (self._use_default, self._mode, self._tee_height)

    def status_text(self) -> str:
        """Return the visible setup explanation."""
        return self._status.text()

    def _apply_setup(self, setup: BallSetup) -> None:
        self._updating = True
        try:
            self._mode.setCurrentIndex(self._mode.findData(setup.support_mode))
            self._tee_height.setValue(setup.tee_height_m * 1000.0)
            if setup.support_mode is BallSupportMode.TEE:
                self._last_tee_height_mm = setup.tee_height_m * 1000.0
        finally:
            self._updating = False
        self._update_enabled_and_status()

    def _mark_override(self) -> None:
        if self._use_default.isChecked():
            self._updating = True
            try:
                self._use_default.setChecked(False)
            finally:
                self._updating = False

    def _on_default_toggled(self, use_default: bool) -> None:
        if self._updating:
            return
        if use_default:
            self._apply_setup(self._club_default)
        else:
            self._update_enabled_and_status()
        self.setupChanged.emit(self.setup())

    def _on_mode_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._mark_override()
        mode = BallSupportMode(self._mode.currentData())
        self._updating = True
        try:
            if mode is BallSupportMode.GROUND:
                if self._tee_height.value() > 0.0:
                    self._last_tee_height_mm = self._tee_height.value()
                self._tee_height.setValue(0.0)
            elif self._tee_height.value() == 0.0:
                self._tee_height.setValue(self._last_tee_height_mm)
        finally:
            self._updating = False
        self._update_enabled_and_status()
        self.setupChanged.emit(self.setup())

    def _on_height_changed(self, height_mm: float) -> None:
        if self._updating:
            return
        self._mark_override()
        self._last_tee_height_mm = height_mm
        self._update_enabled_and_status()
        self.setupChanged.emit(self.setup())

    def _update_enabled_and_status(self) -> None:
        setup = self.setup()
        on_tee = setup.support_mode is BallSupportMode.TEE
        self._tee_height.setEnabled(on_tee)
        if on_tee:
            self._tee_height.setToolTip(_TEE_HEIGHT_GUIDANCE)
            detail = f"Tee, {setup.tee_height_m * 1000.0:.2f} mm to ball bottom."
        else:
            self._tee_height.setToolTip(
                "Ground mode disables Tee Height and enforces 0 mm effective "
                "height. Source: Tools #4143 physical setup contract."
            )
            detail = "Ground mode: the bottom of the ball rests on the ground plane."
        source = (
            f"Using {self._club_name} default."
            if self._use_default.isChecked()
            else "Using explicit user override."
        )
        self._status.setText(f"{source} {detail}")
