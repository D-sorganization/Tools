"""Prescribed joint-torque authoring and profile-library UI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.torque_profile_controller import (
    ProfileDraft,
    RunMode,
    TorqueExecutionSelection,
    TorqueProfileLibraryAdapter,
    execution_selection,
)
from shared.python.signal_toolkit.polynomial_generator import (
    PolynomialGeneratorWidget,
)
from shared.python.swing_sim.torque_profiles import PrescribedTorqueProfile

_MODEL_JOINTS = {
    "model.double_pendulum.v1": (
        ("joint.shoulder", "Shoulder"),
        ("joint.wrist", "Wrist"),
    ),
    "model.triple_pendulum.v1": (
        ("joint.shoulder", "Shoulder"),
        ("joint.wrist", "Wrist"),
        ("joint.club", "Club"),
    ),
}

_MODE_DESCRIPTIONS = {
    RunMode.OPTIMIZED_DEFAULT: (
        "Uses the selected simulator source and its default or solver-configured "
        "motion without applying a prescribed joint-torque profile."
    ),
    RunMode.PRESCRIBED_TORQUE: (
        "Executes a complete double-pendulum profile in the time-aware Python "
        "dynamics kernel. Triple-pendulum profiles can be authored and exchanged, "
        "but are not yet executable."
    ),
}


def _clickable(button: QPushButton, tooltip: str) -> QPushButton:
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setToolTip(tooltip)
    return button


class TorquePolynomialDialog(QDialog):
    """Modal host for the shared polynomial generator."""

    polynomialAccepted = pyqtSignal(str, list)  # noqa: N815

    def __init__(self, joint_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Design Torque — {joint_id}")
        self.resize(1000, 760)
        layout = QVBoxLayout(self)
        description = QLabel(
            "Draw, enter, or fit torque versus physical time. Generated "
            "coefficients are stored as [c0, c1, …] in N·m."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.generator = PolynomialGeneratorWidget(
            self,
            use_builtin_theme=False,
            error_handler=self._show_error,
        )
        self.generator.set_joints([joint_id])
        self.generator.polynomial_generated.connect(self._accept_polynomial)
        layout.addWidget(self.generator)

    def _accept_polynomial(self, joint_id: str, coefficients: list[float]) -> None:
        self.polynomialAccepted.emit(joint_id, coefficients)
        self.accept()

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)


class TorqueProfilePanel(QWidget):
    """Author canonical profiles while preserving the existing run path."""

    runModeChanged = pyqtSignal(object)  # noqa: N815
    profileChanged = pyqtSignal(object)  # noqa: N815

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = TorqueProfileLibraryAdapter()
        self._assignment_buttons: dict[str, QPushButton] = {}
        self._assignment_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_library_group())
        layout.addWidget(self._build_details_group())
        layout.addWidget(self._build_assignment_group())
        self._status_label = QLabel(
            "No prescribed profile has been authored or loaded."
        )
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("torqueProfileStatus")
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        self._on_mode_changed()
        self._rebuild_assignment_rows()

    def _build_mode_group(self) -> QGroupBox:
        box = QGroupBox("Run Mode")
        layout = QVBoxLayout(box)
        self._run_mode_combo = QComboBox()
        self._run_mode_combo.addItem(
            "Default / Solver-Configured", RunMode.OPTIMIZED_DEFAULT
        )
        self._run_mode_combo.addItem("Prescribed Torque", RunMode.PRESCRIBED_TORQUE)
        self._run_mode_combo.setToolTip(
            "Choose the current simulator path or stage a canonical prescribed "
            "joint-torque profile."
        )
        self._run_mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self._run_mode_combo)
        self._mode_description = QLabel()
        self._mode_description.setWordWrap(True)
        self._mode_description.setObjectName("runModeDescription")
        layout.addWidget(self._mode_description)
        return box

    def _build_library_group(self) -> QGroupBox:
        box = QGroupBox("Profile Library")
        layout = QVBoxLayout(box)
        self._profile_combo = QComboBox()
        self._profile_combo.setPlaceholderText("No profiles loaded")
        self._profile_combo.setToolTip(
            "Select a canonical prescribed-torque profile from this session."
        )
        self._profile_combo.currentIndexChanged.connect(self._select_profile)
        layout.addWidget(self._profile_combo)
        actions = QGridLayout()
        self._save_library_button = self._action_button(
            "Save Library…", "Save", self._save_library
        )
        self._load_library_button = self._action_button(
            "Load Library…", "Load", self._load_library
        )
        self._import_profile_button = self._action_button(
            "Import Profile…", "Import", self._import_profile
        )
        self._export_profile_button = self._action_button(
            "Export Profile…", "Export", self._export_profile
        )
        actions.addWidget(self._save_library_button, 0, 0)
        actions.addWidget(self._load_library_button, 0, 1)
        actions.addWidget(self._import_profile_button, 1, 0)
        actions.addWidget(self._export_profile_button, 1, 1)
        layout.addLayout(actions)
        return box

    def _action_button(
        self, label: str, verb: str, callback: Callable[[], None]
    ) -> QPushButton:
        button = _clickable(
            QPushButton(label),
            f"{verb} canonical torque-profile JSON without changing its schema.",
        )
        button.clicked.connect(callback)
        return button

    def _build_details_group(self) -> QGroupBox:
        box = QGroupBox("Profile Details")
        form = QFormLayout(box)
        self._profile_id_edit = QLineEdit("profile.rate_of_closure.driver.v1")
        self._name_edit = QLineEdit("Driver Torque Profile")
        self._description_edit = QLineEdit(
            "Prescribed driver-swing joint torques authored in Rate of Closure."
        )
        self._model_combo = QComboBox()
        self._model_combo.addItem("Double Pendulum", "model.double_pendulum.v1")
        self._model_combo.addItem("Triple Pendulum", "model.triple_pendulum.v1")
        self._model_combo.currentIndexChanged.connect(self._rebuild_assignment_rows)
        self._time_start_spin = QDoubleSpinBox()
        self._time_end_spin = QDoubleSpinBox()
        for spin, value in ((self._time_start_spin, 0.0), (self._time_end_spin, 1.5)):
            spin.setRange(-10.0, 60.0)
            spin.setDecimals(4)
            spin.setValue(value)
            spin.setSuffix(" s")
        form.addRow("Profile ID", self._profile_id_edit)
        form.addRow("Name", self._name_edit)
        form.addRow("Description", self._description_edit)
        form.addRow("Model", self._model_combo)
        domain = QHBoxLayout()
        domain.addWidget(self._time_start_spin)
        domain.addWidget(QLabel("to"))
        domain.addWidget(self._time_end_spin)
        form.addRow("Time Domain", domain)
        return box

    def _build_assignment_group(self) -> QGroupBox:
        self._assignment_group = QGroupBox("Joint Assignments")
        self._assignment_layout = QGridLayout(self._assignment_group)
        explanation = QLabel(
            "Each button opens the shared visual polynomial generator for that "
            "stable joint ID."
        )
        explanation.setWordWrap(True)
        self._assignment_layout.addWidget(explanation, 0, 0, 1, 3)
        return self._assignment_group

    def _rebuild_assignment_rows(self) -> None:
        while self._assignment_layout.count() > 1:
            item = self._assignment_layout.takeAt(1)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._assignment_buttons = {}
        self._assignment_labels = {}
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
        for row, (joint_id, label) in enumerate(_MODEL_JOINTS[model_id], start=1):
            status = QLabel(self._format_coefficients(assigned.get(joint_id)))
            button = _clickable(
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
        dialog = TorquePolynomialDialog(joint_id, self)
        dialog.polynomialAccepted.connect(self.accept_polynomial)
        dialog.exec()

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
        self.profileChanged.emit(profile)

    def _on_mode_changed(self) -> None:
        mode = self._run_mode_combo.currentData()
        self._mode_description.setText(_MODE_DESCRIPTIONS[mode])
        self.runModeChanged.emit(mode)

    def selection(self) -> TorqueExecutionSelection:
        """Return the validated selection consumed by simulation execution."""
        return execution_selection(
            self._run_mode_combo.currentData(), self._library.active_profile()
        )

    def canonical_library(self):  # type: ignore[no-untyped-def]
        """Return the immutable shared profile library for execution."""
        return self._library.canonical_library()

    def set_execution_status(self, message: str) -> None:
        """Show a simulation execution result beside the authoring controls."""
        self._status_label.setText(message)

    def assignment_buttons(self) -> dict[str, QPushButton]:
        """Return a copy of stable joint IDs to visible edit entry points."""
        return dict(self._assignment_buttons)

    def assignment_status(self, joint_id: str) -> str:
        """Return the visible polynomial summary for a stable joint ID."""
        return self._assignment_labels[joint_id].text()

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
            index = self._profile_combo.findData(selected_id)
            self._profile_combo.setCurrentIndex(index)
        self._profile_combo.blockSignals(False)

    def _select_profile(self) -> None:
        profile_id = self._profile_combo.currentData()
        if profile_id is None:
            return
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
        selected = QFileDialog.getExistingDirectory(self, "Save Torque Profile Library")
        if selected:
            self._run_action(
                lambda: self._library.save_library(Path(selected)), "Library saved."
            )

    def _load_library(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Load Torque Profile Library")
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
            self, "Import Torque Profile", "", "JSON Files (*.json)"
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
            self,
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
        QMessageBox.warning(self, title, message)


__all__ = ["TorquePolynomialDialog", "TorqueProfilePanel"]
