"""Prescribed joint-torque authoring and profile-library UI."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.torque_profile_controller import (
    RunMode,
    TorqueProfileLibraryAdapter,
)
from rate_of_closure.ui.pyqt6.torque_profile_dialog import TorquePolynomialDialog
from rate_of_closure.ui.pyqt6.torque_profile_panel_behavior import (
    TorqueProfilePanelBehaviorMixin,
)
from rate_of_closure.ui.pyqt6.torque_profile_widgets import clickable_button


class TorqueProfilePanel(TorqueProfilePanelBehaviorMixin, QWidget):
    """Author canonical profiles while preserving the existing run path."""

    runModeChanged = pyqtSignal(object)  # noqa: N815
    profileChanged = pyqtSignal(object)  # noqa: N815
    jointLocksChanged = pyqtSignal(object)  # noqa: N815
    fitCurrentRunRequested = pyqtSignal(int)  # noqa: N815

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = TorqueProfileLibraryAdapter()
        self._assignment_buttons: dict[str, QPushButton] = {}
        self._assignment_labels: dict[str, QLabel] = {}
        self._joint_lock_checks: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_library_group())
        layout.addWidget(self._build_details_group())
        layout.addWidget(self._build_assignment_group())
        layout.addWidget(self._build_fit_group())
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
        button: QPushButton = clickable_button(
            QPushButton(label),
            f"{verb} canonical torque-profile JSON without changing its schema.",
        )
        button.clicked.connect(callback)
        return button

    def _build_details_group(self) -> QGroupBox:
        box = QGroupBox("Profile Details")
        form = QFormLayout(box)
        self._profile_id_edit = QLineEdit("profile.rate_of_closure.driver.v1")
        self._profile_id_edit.setToolTip(
            "Stable machine-readable identifier written to canonical torque-profile "
            "JSON. Keep it unique and change it when creating a distinct reusable "
            "profile."
        )
        self._name_edit = QLineEdit("Driver Torque Profile")
        self._name_edit.setToolTip(
            "Human-readable profile name shown in the persistent library and exports."
        )
        self._description_edit = QLineEdit(
            "Prescribed driver-swing joint torques authored in Rate of Closure."
        )
        self._description_edit.setToolTip(
            "Describe the profile's purpose, provenance, assumptions, or intended use."
        )
        self._model_combo = QComboBox()
        self._model_combo.addItem("Double Pendulum", "model.double_pendulum.v1")
        self._model_combo.addItem("Triple Pendulum", "model.triple_pendulum.v1")
        self._model_combo.setToolTip(
            "Select the model-specific joint schema. Double-pendulum profiles can be "
            "executed here; triple-pendulum profiles can be authored and exchanged "
            "but are not yet executable in this workbench."
        )
        self._model_combo.currentIndexChanged.connect(self._rebuild_assignment_rows)
        self._time_start_spin = QDoubleSpinBox()
        self._time_end_spin = QDoubleSpinBox()
        for spin, value in ((self._time_start_spin, 0.0), (self._time_end_spin, 1.5)):
            spin.setRange(-10.0, 60.0)
            spin.setDecimals(4)
            spin.setValue(value)
            spin.setSuffix(" s")
        self._time_start_spin.setToolTip(
            "Start of the polynomial torque profile's physical time domain, in "
            "seconds. The domain must cover the complete simulation when executing "
            "the profile."
        )
        self._time_end_spin.setToolTip(
            "End of the polynomial torque profile's physical time domain, in "
            "seconds. It must be later than the start and cover the complete "
            "simulation when executing the profile."
        )
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
            "stable joint ID. Lock Motion applies an ideal zero-velocity constraint "
            "to that coordinate during double-pendulum execution."
        )
        explanation.setWordWrap(True)
        self._assignment_layout.addWidget(explanation, 0, 0, 1, 4)
        return self._assignment_group

    def _build_fit_group(self) -> QGroupBox:
        box = QGroupBox("Fit Retained Run Torques")
        layout = QHBoxLayout(box)
        degree_label = QLabel("Degree")
        degree_label.setToolTip(
            "Polynomial degree used for each retained joint-torque history. Degree 3 "
            "is a compact default; lower degrees are smoother and easier to interpret."
        )
        layout.addWidget(degree_label)
        self._fit_degree_spin = QSpinBox()
        self._fit_degree_spin.setRange(0, 8)
        self._fit_degree_spin.setValue(3)
        self._fit_degree_spin.setToolTip(
            "Fit degree for each joint's applied torque versus physical time. The "
            "fit is rejected if its numerical conditioning is unsafe."
        )
        layout.addWidget(self._fit_degree_spin)
        self._fit_current_run_button = clickable_button(
            QPushButton("Fit Current Run to Profile"),
            "Fit the current double-pendulum run's retained torque history into "
            "a reusable canonical profile, preserving fit quality and provenance.",
        )
        self._fit_current_run_button.clicked.connect(
            lambda: self.fitCurrentRunRequested.emit(self._fit_degree_spin.value())
        )
        layout.addWidget(self._fit_current_run_button, stretch=1)
        return box


__all__ = ["TorquePolynomialDialog", "TorqueProfilePanel"]
