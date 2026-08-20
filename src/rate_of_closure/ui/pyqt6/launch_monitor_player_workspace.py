"""Explicit-identity player analytics controls for the PyQt workspace."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.launch_monitor_analysis import (
    AnalysisRequest,
    AnalysisResult,
    analyze_launch_monitor_data,
    numeric_columns,
)
from rate_of_closure.launch_monitor_workspace import (
    AnalysisSelection,
    LaunchMonitorProject,
    PlayerIdentityBinding,
    dataset_reference_for_frame,
    export_analysis_bundle,
    load_project,
    save_project,
)


class LaunchMonitorPlayerWorkspace(QWidget):
    """Run delegated per-player covariation with explicit identity consent."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._source_name = "Unloaded"
        self._result: AnalysisResult | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.identity_combo = QComboBox()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.attestation = QCheckBox(
            "I attest this column identifies a player; it was not inferred "
            "from session, club, or row order."
        )
        self.run_button = QPushButton("Run Offline Compatibility Covariation")
        self.save_button = QPushButton("Save Project...")
        self.load_button = QPushButton("Load Project...")
        self.export_button = QPushButton("Export Full Bundle...")
        self.status = QLabel("Select and attest an explicit player identity column.")
        self.status.setWordWrap(True)
        descriptions = (
            (self.identity_combo, "Player identity column"),
            (self.x_combo, "Player covariation X variable"),
            (self.y_combo, "Player covariation Y variable"),
            (self.attestation, "Explicit player identity attestation"),
            (
                self.run_button,
                "Run local v1 compatibility calculation; canonical v2 is preferred",
            ),
            (
                self.save_button,
                "Save reference-only project; backing rows are not embedded",
            ),
            (self.load_button, "Load a saved reference-only project"),
            (self.export_button, "Export project, result, and explicit backing rows"),
        )
        for control, description in descriptions:
            control.setAccessibleName(description)
            control.setToolTip(description)
        form = QFormLayout()
        form.addRow("Player identity:", self.identity_combo)
        form.addRow("X variable:", self.x_combo)
        form.addRow("Y variable:", self.y_combo)
        form.addRow(self.attestation)
        buttons = QHBoxLayout()
        for button in (
            self.run_button,
            self.save_button,
            self.load_button,
            self.export_button,
        ):
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Player Covariation Workspace"))
        boundary = QLabel(
            "Canonical UpstreamDrift v2 is preferred. This standalone run is an "
            "explicit local v1 compatibility/offline calculation; row-aligned "
            "residuals are unavailable."
        )
        boundary.setWordWrap(True)
        boundary.setAccessibleName("Player analytics authority status")
        boundary.setToolTip(
            "Explains why this embedded calculation is not a canonical v2 response."
        )
        layout.addWidget(boundary)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        self.attestation.toggled.connect(self._refresh_enabled)
        self.identity_combo.currentTextChanged.connect(self._refresh_enabled)
        self.run_button.clicked.connect(self.run_safely)
        self.save_button.clicked.connect(self.save_dialog)
        self.load_button.clicked.connect(self.load_dialog)
        self.export_button.clicked.connect(self.export_dialog)
        self._refresh_enabled()

    def set_dataset(self, frame: pd.DataFrame, source_name: str) -> None:
        """Replace the referenced rows and reset identity attestation."""

        self._frame = frame
        self._source_name = source_name
        string_columns = sorted(str(column) for column in frame.columns)
        numbers = numeric_columns(frame)
        for combo, values in (
            (self.identity_combo, string_columns),
            (self.x_combo, numbers),
            (self.y_combo, numbers),
        ):
            combo.clear()
            combo.addItem("")
            combo.addItems(values)
        if "face_angle" in numbers:
            self.x_combo.setCurrentText("face_angle")
        if "club_path" in numbers:
            self.y_combo.setCurrentText("club_path")
        self.attestation.setChecked(False)
        self._result = None
        self.status.setText("Select and attest an explicit player identity column.")
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        ready = bool(self.identity_combo.currentText()) and self.attestation.isChecked()
        self.run_button.setEnabled(ready)
        self.save_button.setEnabled(ready)
        self.export_button.setEnabled(ready and self._result is not None)

    def project(self) -> LaunchMonitorProject:
        """Return validated reference-only state for the current controls."""

        return LaunchMonitorProject(
            name=f"{self._source_name} player covariation",
            dataset=dataset_reference_for_frame(self._frame, self._source_name),
            identity=PlayerIdentityBinding(
                self.identity_combo.currentText(), self.attestation.isChecked()
            ),
            selection=AnalysisSelection(
                self.x_combo.currentText(), self.y_combo.currentText()
            ),
        )

    def run_player_analysis(self) -> AnalysisResult:
        """Delegate per-player estimates to the existing analysis authority seam."""

        project = self.project()
        result = analyze_launch_monitor_data(
            self._frame,
            AnalysisRequest(
                outcome=project.selection.y,
                predictors=(project.selection.x,),
                analysis_mode="correlation",
                group_by=project.identity.column,
                min_samples=project.selection.min_samples,
                confidence_level=project.selection.confidence_level,
            ),
        )
        self._result = result
        group_count = len(result.groups)
        self.status.setText(
            f"{group_count} player groups analyzed. Associations are not causal; "
            "no player identity was inferred."
        )
        self._refresh_enabled()
        return result

    def run_safely(self) -> None:
        try:
            self.run_player_analysis()
        except ValueError as error:
            QMessageBox.warning(self, "Player Analysis Not Run", str(error))

    def save_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Launch Monitor Project",
            "analysis.lmproject.json",
            "JSON (*.json)",
        )
        if selected:
            save_project(selected, self.project())

    def load_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Load Launch Monitor Project", "", "JSON (*.json)"
        )
        if not selected:
            return
        try:
            project = load_project(selected)
            current = dataset_reference_for_frame(self._frame, self._source_name)
            if project.dataset.sha256 != current.sha256:
                raise ValueError("saved project references a different dataset")
            self.identity_combo.setCurrentText(project.identity.column)
            self.x_combo.setCurrentText(project.selection.x)
            self.y_combo.setCurrentText(project.selection.y)
            self.attestation.setChecked(project.identity.user_attested)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Project Not Loaded", str(error))

    def export_dialog(self) -> None:
        if self._result is None:
            return
        selected = QFileDialog.getExistingDirectory(self, "Choose Full Export Parent")
        if selected:
            export_analysis_bundle(
                Path(selected) / "launch-monitor-analysis-bundle",
                self.project(),
                self._result.to_wire(),
                self._frame,
            )


__all__ = ["LaunchMonitorPlayerWorkspace"]
