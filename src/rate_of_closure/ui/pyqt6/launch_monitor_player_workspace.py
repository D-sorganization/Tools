"""Explicit-identity player analytics controls for the PyQt workspace."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.launch_monitor_analysis import (
    AnalysisRequest,
    AnalysisResult,
    analyze_launch_monitor_data,
    numeric_columns,
)
from rate_of_closure.launch_monitor_v2_client import (
    MAX_CANONICAL_INLINE_RECORDS,
    CanonicalDatasetReference,
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
from rate_of_closure.player_covariation import (
    CovariationRequest,
    PairScanRequest,
    PlayerCovariationAnalysis,
    analyze_player_covariation,
    scan_covariation_pairs,
)
from rate_of_closure.ui.pyqt6.launch_monitor_canonical_workspace_mixin import (
    CanonicalWorkspaceMixin,
)
from rate_of_closure.ui.pyqt6.launch_monitor_covariation_view import (
    LaunchMonitorCovariationView,
)


class LaunchMonitorPlayerWorkspace(CanonicalWorkspaceMixin, QWidget):
    """Run delegated per-player covariation with explicit identity consent."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._source_name = "Unloaded"
        self._result: AnalysisResult | None = None
        self.covariation_result: PlayerCovariationAnalysis | None = None
        self._export_payload: dict[str, object] = {}
        self._canonical_reference: CanonicalDatasetReference | None = None
        self._canonical_job_id: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.identity_combo = QComboBox()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Pearson", "Spearman"])
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(4, 1_000_000)
        self.min_samples_spin.setValue(10)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.51, 0.999)
        self.confidence_spin.setDecimals(3)
        self.confidence_spin.setValue(0.95)
        self.attestation = QCheckBox(
            "I attest this column identifies a player; it was not inferred "
            "from session, club, or row order."
        )
        self._build_canonical_controls()
        self.run_button = QPushButton("Run Offline Compatibility Covariation")
        self.scan_button = QPushButton("Rank Variable Pairs")
        self.save_button = QPushButton("Save Project...")
        self.load_button = QPushButton("Load Project...")
        self.export_button = QPushButton("Export Full Bundle...")
        self.status = QLabel("Select and attest an explicit player identity column.")
        self.status.setWordWrap(True)
        descriptions = (
            (self.identity_combo, "Player identity column"),
            (self.x_combo, "Player covariation X variable"),
            (self.y_combo, "Player covariation Y variable"),
            (self.method_combo, "Displayed covariation coefficient"),
            (self.min_samples_spin, "Minimum pairwise-complete shots per player"),
            (self.confidence_spin, "Pearson confidence level"),
            (self.attestation, "Explicit player identity attestation"),
            (self.authority_url, "Canonical Upstream authority URL"),
            (self.corpus_reference_button, "Load authorized corpus reference"),
            (self.inspect_corpus_button, "Inspect authorized corpus aggregates"),
            (self.refresh_corpus_button, "Refresh canonical corpus job"),
            (
                self.canonical_covariation_button,
                "Run canonical Upstream player covariation for at most 20,000 rows",
            ),
            (
                self.run_button,
                "Run local v1 compatibility calculation; canonical v2 is preferred",
            ),
            (
                self.scan_button,
                "Exploratory scan of all numeric pairs with multiplicity warning",
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
        form.addRow("Coefficient:", self.method_combo)
        form.addRow("Minimum N/player:", self.min_samples_spin)
        form.addRow("Confidence:", self.confidence_spin)
        form.addRow(self.attestation)
        form.addRow("Canonical authority:", self.authority_url)
        form.addRow(self.corpus_reference_button)
        form.addRow(self.canonical_limit)
        buttons = QGridLayout()
        for index, button in enumerate(
            (
                self.inspect_corpus_button,
                self.refresh_corpus_button,
                self.canonical_covariation_button,
                self.run_button,
                self.scan_button,
                self.save_button,
                self.load_button,
                self.export_button,
            )
        ):
            buttons.addWidget(button, index // 2, index % 2)
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
        self.covariation_view = LaunchMonitorCovariationView()
        layout.addWidget(self.covariation_view)
        self.attestation.toggled.connect(self._refresh_enabled)
        self.identity_combo.currentTextChanged.connect(self._refresh_enabled)
        self.run_button.clicked.connect(self.run_safely)
        self.scan_button.clicked.connect(self.scan_safely)
        self.save_button.clicked.connect(self.save_dialog)
        self.load_button.clicked.connect(self.load_dialog)
        self.export_button.clicked.connect(self.export_dialog)
        self.corpus_reference_button.clicked.connect(self.load_corpus_reference_dialog)
        self.inspect_corpus_button.clicked.connect(self.submit_corpus_job_safely)
        self.refresh_corpus_button.clicked.connect(self.refresh_corpus_job_safely)
        self.canonical_covariation_button.clicked.connect(
            self.run_canonical_covariation_safely
        )
        self.authority_url.textChanged.connect(self._refresh_enabled)
        self._refresh_enabled()

    def set_dataset(
        self,
        frame: pd.DataFrame,
        source_name: str,
        numeric_fields: list[str] | None = None,
    ) -> None:
        """Replace the referenced rows and reset identity attestation."""
        self._frame = frame
        self._source_name = source_name
        string_columns = sorted(str(column) for column in frame.columns)
        numbers = numeric_columns(frame) if numeric_fields is None else numeric_fields
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
        self.covariation_result = None
        self._export_payload = {}
        self.status.setText("Select and attest an explicit player identity column.")
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        ready = bool(self.identity_combo.currentText()) and self.attestation.isChecked()
        authority_ready = bool(self.authority_url.text().strip())
        self.run_button.setEnabled(ready)
        self.scan_button.setEnabled(ready)
        self.save_button.setEnabled(ready)
        self.export_button.setEnabled(ready and bool(self._export_payload))
        self.inspect_corpus_button.setEnabled(
            authority_ready and self._canonical_reference is not None
        )
        self.refresh_corpus_button.setEnabled(
            authority_ready and self._canonical_job_id is not None
        )
        self.canonical_covariation_button.setEnabled(
            ready
            and authority_ready
            and 0 < len(self._frame) <= MAX_CANONICAL_INLINE_RECORDS
        )

    def project(self) -> LaunchMonitorProject:
        """Return validated reference-only state for the current controls."""
        return LaunchMonitorProject(
            name=f"{self._source_name} player covariation",
            dataset=dataset_reference_for_frame(self._frame, self._source_name),
            identity=PlayerIdentityBinding(
                self.identity_combo.currentText(), self.attestation.isChecked()
            ),
            selection=AnalysisSelection(
                self.x_combo.currentText(),
                self.y_combo.currentText(),
                self.min_samples_spin.value(),
                self.confidence_spin.value(),
            ),
            canonical_dataset=self._canonical_reference,
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
        self.covariation_result = analyze_player_covariation(
            self._frame,
            CovariationRequest(
                x_column=project.selection.x,
                y_column=project.selection.y,
                player_column=project.identity.column,
                min_samples=project.selection.min_samples,
                confidence_level=project.selection.confidence_level,
            ),
        )
        self._export_payload = self._covariation_payload(self.covariation_result)
        self.covariation_view.populate(self.covariation_result)
        group_count = len(result.groups)
        self.status.setText(
            f"{group_count} player groups analyzed. Associations are not causal; "
            "no player identity was inferred."
        )
        self._refresh_enabled()
        return result

    def run_pair_scan(self) -> None:
        """Rank all numeric pairs using the same attested player identity."""

        project = self.project()
        analysis = scan_covariation_pairs(
            self._frame,
            PairScanRequest(
                player_column=project.identity.column,
                numeric_columns=tuple(numeric_columns(self._frame)),
                min_samples=project.selection.min_samples,
                confidence_level=project.selection.confidence_level,
            ),
        )
        self._export_payload = {
            "contract_version": "player-covariation-scan/1.0.0",
            "mode": "exploratory_pair_scan",
            "request": {
                "player_column": project.identity.column,
                "numeric_columns": list(numeric_columns(self._frame)),
                "min_samples": project.selection.min_samples,
                "confidence_level": project.selection.confidence_level,
            },
            "ranking": analysis.ranking.to_dict(orient="records"),
            "warnings": list(analysis.warnings),
            "method_description": analysis.method_description,
        }
        self.covariation_view.populate_scan(analysis.ranking)
        self.status.setText(
            f"{len(analysis.ranking)} exploratory pairs ranked. Multiple-comparison "
            "control or held-out confirmation is required."
        )

    def scan_safely(self) -> None:
        try:
            self.run_pair_scan()
        except ValueError as error:
            QMessageBox.warning(self, "Pair Scan Not Run", str(error))

    @staticmethod
    def _covariation_payload(
        analysis: PlayerCovariationAnalysis,
    ) -> dict[str, object]:
        return {
            "contract_version": "player-covariation/1.0.0",
            "request": asdict(analysis.request),
            "units": analysis.units,
            "definitions": analysis.definitions,
            "pooled": asdict(analysis.pooled),
            "within_player": asdict(analysis.within_player),
            "between_player": asdict(analysis.between_player),
            "meta_analysis": asdict(analysis.meta_analysis),
            "per_player": analysis.per_player.to_dict(orient="records"),
            "backing_data": analysis.backing_data.to_dict(orient="records"),
            "warnings": list(analysis.warnings),
            "method_description": analysis.method_description,
        }

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
            self.min_samples_spin.setValue(project.selection.min_samples)
            self.confidence_spin.setValue(project.selection.confidence_level)
            self.attestation.setChecked(project.identity.user_attested)
            self._canonical_reference = project.canonical_dataset
            self._refresh_enabled()
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Project Not Loaded", str(error))

    def export_dialog(self) -> None:
        if not self._export_payload:
            return
        selected = QFileDialog.getExistingDirectory(self, "Choose Full Export Parent")
        if selected:
            export_analysis_bundle(
                Path(selected) / "launch-monitor-analysis-bundle",
                self.project(),
                self._export_payload,
                self._frame,
            )


__all__ = ["LaunchMonitorPlayerWorkspace"]
