"""Behavior mixin for the prescribed joint-torque profile panel."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from rate_of_closure.simulation.records import SimulationRun
from rate_of_closure.ui.pyqt6.torque_profile_controller import (
    ProfileDraft,
    TorqueExecutionSelection,
    TorqueProfileLibraryAdapter,
    execution_selection,
)
from rate_of_closure.ui.pyqt6.torque_profile_dialog import TorquePolynomialDialog
from rate_of_closure.ui.pyqt6.torque_profile_widgets import (
    MODE_DESCRIPTIONS,
    MODEL_JOINTS,
    clickable_button,
)
from shared.python.swing_sim.run_config import (
    DOUBLE_PENDULUM_MODEL_ID,
    JointLockConfig,
)
from shared.python.swing_sim.torque_library import TorqueProfileLibrary
from shared.python.swing_sim.torque_profiles import PrescribedTorqueProfile


class TorqueProfilePanelBehaviorMixin:
    """Own profile assignment, execution selection, persistence, and status."""

    if TYPE_CHECKING:
        _library: TorqueProfileLibraryAdapter
        _assignment_buttons: dict[str, QPushButton]
        _assignment_labels: dict[str, QLabel]
        _joint_lock_checks: dict[str, QCheckBox]
        _assignment_layout: QGridLayout
        _profile_id_edit: QLineEdit
        _name_edit: QLineEdit
        _description_edit: QLineEdit
        _model_combo: QComboBox
        _time_start_spin: QDoubleSpinBox
        _time_end_spin: QDoubleSpinBox
        _run_mode_combo: QComboBox
        _mode_description: QLabel
        _status_label: QLabel
        _profile_combo: QComboBox
        _fit_degree_spin: QSpinBox
        _fit_current_run_button: QPushButton
        _save_library_button: QPushButton
        _load_library_button: QPushButton
        _import_profile_button: QPushButton
        _export_profile_button: QPushButton

        def _emit_profile_changed(self, profile: object) -> None: ...

        def _emit_run_mode_changed(self, mode: object) -> None: ...

        def _emit_joint_locks_changed(self, locks: object) -> None: ...

    def _rebuild_assignment_rows(self) -> None:
        retained_locks = self.joint_locks().locked_joint_ids
        while self._assignment_layout.count() > 1:
            item = self._assignment_layout.takeAt(1)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._assignment_buttons = {}
        self._assignment_labels = {}
        self._joint_lock_checks = {}
        model_id = str(self._model_combo.currentData())
        profile = self._library.active_profile()
        assigned = (
            {}
            if profile is None
            else {
                item.joint_id: item.polynomial.coefficients
                for item in profile.assignments
                if profile.model_id == model_id
            }
        )
        for row, (joint_id, label) in enumerate(MODEL_JOINTS[model_id], start=1):
            status = QLabel(self._format_coefficients(assigned.get(joint_id)))
            button = clickable_button(
                QPushButton("Assign / Edit…"),
                f"Open the shared polynomial generator and assign {joint_id}.",
            )
            button.clicked.connect(
                lambda _checked=False, selected=joint_id: self._open_editor(selected)
            )
            self._assignment_layout.addWidget(QLabel(label), row, 0)
            self._assignment_layout.addWidget(status, row, 1)
            self._assignment_layout.addWidget(button, row, 2)
            self._assignment_labels[joint_id] = status
            self._assignment_buttons[joint_id] = button
            self._add_joint_lock_control(row, joint_id, retained_locks)
        if model_id != DOUBLE_PENDULUM_MODEL_ID and retained_locks:
            self._on_joint_locks_changed()

    def _add_joint_lock_control(
        self, row: int, joint_id: str, retained_locks: tuple[str, ...]
    ) -> None:
        if str(self._model_combo.currentData()) != DOUBLE_PENDULUM_MODEL_ID:
            unavailable = QLabel("Lock unavailable")
            unavailable.setToolTip(
                "Ideal joint locks currently execute only in the "
                "double-pendulum dynamics kernel."
            )
            self._assignment_layout.addWidget(unavailable, row, 3)
            return
        lock = QCheckBox("Lock Motion")
        coordinate = (
            "shoulder coordinate theta1, an absolute angle measured relative "
            "to the fixed ground frame"
            if joint_id == "joint.shoulder"
            else "wrist coordinate theta2, a relative angle measured from the "
            "upper segment to the club segment"
        )
        lock.setToolTip(
            f"Apply an ideal zero-velocity constraint to the {coordinate} "
            f"({joint_id}). The coordinate remains at its initial angle; "
            "constraint reaction torque is reported separately from commanded torque."
        )
        lock.setChecked(joint_id in retained_locks)
        lock.toggled.connect(self._on_joint_locks_changed)
        self._assignment_layout.addWidget(lock, row, 3)
        self._joint_lock_checks[joint_id] = lock

    def _draft(self) -> ProfileDraft:
        return ProfileDraft(
            profile_id=self._profile_id_edit.text().strip(),
            model_id=str(self._model_combo.currentData()),
            name=self._name_edit.text().strip(),
            description=self._description_edit.text().strip(),
            time_domain_s=(
                self._time_start_spin.value(),
                self._time_end_spin.value(),
            ),
        )

    def _open_editor(self, joint_id: str) -> None:
        dialog = TorquePolynomialDialog(joint_id, self._panel_widget())
        dialog.polynomialAccepted.connect(self.accept_polynomial)
        dialog.exec()

    def _panel_widget(self) -> QWidget:
        if not isinstance(self, QWidget):
            raise TypeError("torque profile behavior host must be a QWidget")
        return self

    def accept_polynomial(self, joint_id: str, coefficients: list[float]) -> None:
        """Accept c0-first coefficients emitted by the shared generator."""
        try:
            profile = self._library.assign(self._draft(), joint_id, coefficients)
        except (KeyError, TypeError, ValueError) as error:
            self._show_error("Invalid Torque Profile", str(error))
            return
        self._refresh_profiles(profile.profile_id)
        self._assignment_labels[joint_id].setText(
            self._format_coefficients(tuple(coefficients))
        )
        self._status_label.setText(
            f"Staged {profile.name}: {len(profile.assignments)} joint assignment(s)."
        )
        self._emit_profile_changed(profile)

    def _on_mode_changed(self) -> None:
        mode = self._run_mode_combo.currentData()
        self._mode_description.setText(MODE_DESCRIPTIONS[mode])
        self._emit_run_mode_changed(mode)

    def selection(self) -> TorqueExecutionSelection:
        """Return the validated selection consumed by simulation execution."""
        return execution_selection(
            self._run_mode_combo.currentData(), self._library.active_profile()
        )

    def joint_locks(self) -> JointLockConfig:
        """Return ideal constraints in stable double-pendulum joint order."""
        return JointLockConfig(
            tuple(
                joint_id
                for joint_id, checkbox in self._joint_lock_checks.items()
                if checkbox.isChecked()
            )
        )

    def joint_lock_summary(self) -> str:
        """Return a concise human-readable ideal-constraint summary."""
        locks = self.joint_locks().locked_joint_ids
        if not locks:
            return "All joints free"
        labels = dict(MODEL_JOINTS[DOUBLE_PENDULUM_MODEL_ID])
        return ", ".join(f"{labels[joint_id]} locked" for joint_id in locks)

    def clear_joint_locks(self, *, emit: bool = True) -> None:
        """Clear ideal constraints, optionally notifying the simulation host."""
        changed = False
        for checkbox in self._joint_lock_checks.values():
            if checkbox.isChecked():
                changed = True
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
        if not changed:
            return
        self._status_label.setText("Joint constraints updated: All joints free.")
        if emit:
            self._emit_joint_locks_changed(JointLockConfig())

    def _on_joint_locks_changed(self, *_args: object) -> None:
        locks = self.joint_locks()
        self._status_label.setText(
            f"Joint constraints updated: {self.joint_lock_summary()}."
        )
        self._emit_joint_locks_changed(locks)

    def fit_current_run(
        self, run: SimulationRun, degree: int | None = None
    ) -> PrescribedTorqueProfile | None:
        """Fit retained run torques into the active canonical profile draft."""
        selected_degree = self._fit_degree_spin.value() if degree is None else degree
        try:
            profile = self._library.fit_run(self._draft(), run, selected_degree)
        except (TypeError, ValueError) as error:
            self.set_fit_error(str(error))
            return None
        self._refresh_profiles(profile.profile_id)
        self._display_profile(profile)
        self._status_label.setText(
            f"Fitted {len(profile.assignments)} joint histories at degree "
            f"{selected_degree} and staged {profile.name}."
        )
        self._emit_profile_changed(profile)
        return profile

    def set_fit_error(self, message: str) -> None:
        """Present a non-modal, actionable retained-history fit error."""
        self._status_label.setText(f"Cannot fit current run — {message}")

    def canonical_library(self) -> TorqueProfileLibrary:
        """Return the immutable shared profile library for execution."""
        return self._library.canonical_library()

    def set_execution_status(self, message: str) -> None:
        """Show a simulation execution result beside the authoring controls."""
        self._status_label.setText(message)

    def assignment_buttons(self) -> dict[str, QPushButton]:
        """Return a copy of stable joint IDs to visible edit entry points."""
        return dict(self._assignment_buttons)

    def joint_lock_checkboxes(self) -> dict[str, QCheckBox]:
        """Return stable joint IDs to the adjacent ideal-lock controls."""
        return dict(self._joint_lock_checks)

    def fit_current_run_button(self) -> QPushButton:
        """Return the explicit retained-history fit action."""
        return self._fit_current_run_button

    def fit_degree(self) -> int:
        """Return the selected retained-history polynomial degree."""
        return int(self._fit_degree_spin.value())

    def assignment_status(self, joint_id: str) -> str:
        """Return the visible polynomial summary for a stable joint ID."""
        return str(self._assignment_labels[joint_id].text())

    def library_action_buttons(self) -> dict[str, QPushButton]:
        """Return the four explicit persistence controls."""
        return {
            "save": self._save_library_button,
            "load": self._load_library_button,
            "import": self._import_profile_button,
            "export": self._export_profile_button,
        }

    @staticmethod
    def _format_coefficients(coefficients: tuple[float, ...] | None) -> str:
        if coefficients is None:
            return "Not Assigned"
        return "c = [" + ", ".join(f"{value:g}" for value in coefficients) + "]"

    def _refresh_profiles(self, selected_id: str | None = None) -> None:
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        for profile in self._library.profiles():
            self._profile_combo.addItem(profile.name, profile.profile_id)
        if selected_id is not None:
            self._profile_combo.setCurrentIndex(
                self._profile_combo.findData(selected_id)
            )
        self._profile_combo.blockSignals(False)

    def _select_profile(self) -> None:
        profile_id = self._profile_combo.currentData()
        if profile_id is not None:
            self._display_profile(self._library.set_active(str(profile_id)))

    def _display_profile(self, profile: PrescribedTorqueProfile) -> None:
        self._profile_id_edit.setText(profile.profile_id)
        self._name_edit.setText(profile.name)
        self._description_edit.setText(profile.description)
        model_index = self._model_combo.findData(profile.model_id)
        if model_index >= 0:
            self._model_combo.setCurrentIndex(model_index)
        self._time_start_spin.setValue(profile.time_domain_s[0])
        self._time_end_spin.setValue(profile.time_domain_s[1])
        self._rebuild_assignment_rows()

    def _save_library(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self._panel_widget(), "Save Torque Profile Library"
        )
        if selected:
            self._run_action(
                lambda: self._library.save_library(Path(selected)), "Library saved."
            )

    def _load_library(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self._panel_widget(), "Load Torque Profile Library"
        )
        if not selected:
            return
        loaded = self._run_action(
            lambda: self._library.load_library(Path(selected)), "Library loaded."
        )
        if loaded:
            self._refresh_profiles()
            profile = self._library.active_profile()
            if profile is not None:
                self._display_profile(profile)

    def _import_profile(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self._panel_widget(), "Import Torque Profile", "", "JSON Files (*.json)"
        )
        if not selected:
            return
        result = self._run_action(
            lambda: self._library.import_profile(Path(selected)), "Profile imported."
        )
        if result:
            profile = self._library.active_profile()
            self._refresh_profiles(None if profile is None else profile.profile_id)
            if profile is not None:
                self._display_profile(profile)

    def _export_profile(self) -> None:
        profile = self._library.active_profile()
        if profile is None:
            self._show_error("No Profile Selected", "Author or import a profile first.")
            return
        selected, _ = QFileDialog.getSaveFileName(
            self._panel_widget(),
            "Export Torque Profile",
            f"{profile.profile_id}.json",
            "JSON Files (*.json)",
        )
        if selected:
            self._run_action(
                lambda: self._library.export_profile(
                    profile.profile_id, Path(selected)
                ),
                "Profile exported.",
            )

    def _run_action(self, action: Callable[[], object], success: str) -> bool:
        try:
            action()
        except (OSError, TypeError, ValueError) as error:
            self._show_error("Torque Profile Error", str(error))
            return False
        self._status_label.setText(success)
        return True

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self._panel_widget(), title, message)


__all__ = ["TorqueProfilePanelBehaviorMixin"]
