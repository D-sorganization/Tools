"""PyQt6 twin of the React Launch Monitor Analytics tab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.launch_monitor_analysis import (
    AnalysisMode,
    AnalysisRequest,
    AnalysisResult,
    CorrelationMethod,
    MissingPolicy,
    analyze_launch_monitor_data,
    numeric_columns,
)
from rate_of_closure.launch_monitor_import import read_launch_monitor_frame
from rate_of_closure.launch_monitor_linked_scatter import MAX_RETAINED_ROWS
from rate_of_closure.ui.pyqt6.launch_monitor_analysis_results import (
    render_analysis_result,
)
from rate_of_closure.ui.pyqt6.launch_monitor_linked_scatter_panel import (
    LaunchMonitorLinkedScatterPanel,
)
from rate_of_closure.ui.pyqt6.launch_monitor_preview import demo_frame
from shared.python.swing_sim.conventions import (
    ConventionId,
    ParameterId,
    convention_registry,
)


class LaunchMonitorAnalyticsTab(QWidget):
    """Import retained records and run arbitrary traceable analyses."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frame = demo_frame()
        self.source_name = "Built-In Demonstration Data"
        self.last_result: AnalysisResult | None = None
        self._build_ui()
        self._refresh_columns()

    def _build_ui(self) -> None:
        heading = QLabel("Launch Monitor Analytics")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        boundary = QLabel(
            "Import CSV or JSON without dropping source columns. Correlations and "
            "fitted models are associations, not causal evidence. TrackMan-Comparable "
            "and Foresight-Comparable are documented frames, not device emulation "
            "or certification."
        )
        boundary.setWordWrap(True)
        boundary.setAccessibleName("Launch Monitor Analytics Scientific Boundary")
        self.source_label = QLabel()
        self.source_label.setWordWrap(True)

        self.import_button = QPushButton("Import Data...")
        self.demo_button = QPushButton("Load Demo")
        self.export_data_button = QPushButton("Export Retained Data...")
        self.export_result_button = QPushButton("Export Analysis...")
        self.export_result_button.setEnabled(False)
        buttons = QHBoxLayout()
        for button in (
            self.import_button,
            self.demo_button,
            self.export_data_button,
            self.export_result_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)

        self.convention_combo = QComboBox()
        self.convention_combo.addItem("App-Native", ConventionId.APP_NATIVE)
        self.convention_combo.addItem(
            "TrackMan-Comparable", ConventionId.TRACKMAN_COMPARABLE
        )
        self.convention_combo.addItem(
            "Foresight-Comparable", ConventionId.FORESIGHT_COMPARABLE
        )
        self.convention_evidence = QLabel()
        self.convention_evidence.setWordWrap(True)
        self.convention_evidence.setOpenExternalLinks(True)
        self.outcome_combo = QComboBox()
        self.predictor_list = QListWidget()
        self.predictor_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["comprehensive", "correlation", "regression"])
        self.method_combo = QComboBox()
        self.method_combo.addItems(["pearson", "spearman", "kendall"])
        self.missing_combo = QComboBox()
        self.missing_combo.addItems(["pairwise", "listwise", "fail"])
        self.group_combo = QComboBox()
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.51, 0.999)
        self.confidence_spin.setDecimals(3)
        self.confidence_spin.setValue(0.95)
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(3, 1_000_000)
        self.min_samples_spin.setValue(10)
        self.run_button = QPushButton("Run Analysis")

        controls = (
            (self.import_button, "Import a CSV or JSON launch-monitor export"),
            (self.demo_button, "Restore the built-in demonstration data"),
            (self.export_data_button, "Export every retained input record"),
            (self.export_result_button, "Export request, results, and lineage"),
            (self.convention_combo, "Interpretation Convention"),
            (self.outcome_combo, "Outcome Variable"),
            (self.predictor_list, "Predictor Variables"),
            (self.mode_combo, "Analysis Mode"),
            (self.method_combo, "Correlation Method"),
            (self.missing_combo, "Missing-Data Policy"),
            (self.group_combo, "Optional Grouping Variable"),
            (self.confidence_spin, "Confidence Level"),
            (self.min_samples_spin, "Minimum Sample Count"),
            (self.run_button, "Run the selected statistical analysis"),
        )
        for control, description in controls:
            control.setAccessibleName(description)
            control.setToolTip(description)

        form = QFormLayout()
        form.addRow("Convention:", self.convention_combo)
        form.addRow(self.convention_evidence)
        form.addRow("Outcome:", self.outcome_combo)
        form.addRow("Predictors:", self.predictor_list)
        form.addRow("Analysis Mode:", self.mode_combo)
        form.addRow("Correlation:", self.method_combo)
        form.addRow("Missing Data:", self.missing_combo)
        form.addRow("Group By:", self.group_combo)
        form.addRow("Confidence:", self.confidence_spin)
        form.addRow("Minimum N:", self.min_samples_spin)
        form.addRow(self.run_button)

        self.preview_panel = LaunchMonitorLinkedScatterPanel()
        self.preview = self.preview_panel.preview
        self.preview_status = self.preview_panel.status
        self.result_table = QTableWidget()
        self.result_table.setAccessibleName("Launch Monitor Statistical Results")
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setAccessibleName("Launch Monitor Analysis Traceability")
        output = QSplitter(Qt.Orientation.Vertical)
        output.addWidget(self.preview_panel)
        output.addWidget(self.result_table)
        output.addWidget(self.details)
        output.setSizes([320, 240, 160])

        body = QSplitter(Qt.Orientation.Horizontal)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.addLayout(form)
        controls_layout.addStretch(1)
        body.addWidget(controls_widget)
        body.addWidget(output)
        body.setSizes([360, 900])

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(boundary)
        layout.addLayout(buttons)
        layout.addWidget(self.source_label)
        layout.addWidget(body, 1)

        self.import_button.clicked.connect(self.import_dialog)
        self.demo_button.clicked.connect(self.load_demo)
        self.export_data_button.clicked.connect(self.export_data_dialog)
        self.export_result_button.clicked.connect(self.export_result_dialog)
        self.run_button.clicked.connect(self.run_analysis_safely)
        self.convention_combo.currentIndexChanged.connect(
            self._refresh_convention_evidence
        )
        self.outcome_combo.currentTextChanged.connect(self._refresh_convention_evidence)
        self.outcome_combo.currentTextChanged.connect(self._refresh_preview)
        self.predictor_list.itemSelectionChanged.connect(self._refresh_preview)
        for signal in (
            self.outcome_combo.currentTextChanged,
            self.predictor_list.itemSelectionChanged,
            self.mode_combo.currentTextChanged,
            self.method_combo.currentTextChanged,
            self.missing_combo.currentTextChanged,
            self.group_combo.currentTextChanged,
            self.confidence_spin.valueChanged,
            self.min_samples_spin.valueChanged,
        ):
            signal.connect(self._invalidate_analysis)

    def _invalidate_analysis(self) -> None:
        """Prevent stale request results from remaining visible or exportable."""
        if self.last_result is None:
            return
        self.last_result = None
        self.export_result_button.setEnabled(False)
        self.result_table.clearContents()
        self.result_table.setRowCount(0)
        self.details.setPlainText(
            "Analysis contract changed. Run Analysis to refresh results."
        )

    def _refresh_columns(self) -> None:
        numeric = numeric_columns(self.frame)
        self.outcome_combo.clear()
        self.outcome_combo.addItems(numeric)
        outcome = "ball_speed" if "ball_speed" in numeric else numeric[0]
        self.outcome_combo.setCurrentText(outcome)
        self.predictor_list.clear()
        self.predictor_list.addItems(numeric)
        defaults = {"club_speed", "attack_angle"} - {outcome}
        if not defaults.intersection(numeric):
            defaults = {next(column for column in numeric if column != outcome)}
        for index in range(self.predictor_list.count()):
            item = self.predictor_list.item(index)
            if item is not None:
                item.setSelected(item.text() in defaults)
        groups = sorted(
            str(column)
            for column in self.frame.columns
            if self.frame[column].notna().any()
            and self.frame[column].nunique(dropna=True) <= 100
        )
        self.group_combo.clear()
        self.group_combo.addItem("(none)")
        self.group_combo.addItems(groups)
        if "monitor_vendor" in groups:
            self.group_combo.setCurrentText("monitor_vendor")
        self.source_label.setText(
            f"Source: {self.source_name} · {len(self.frame)} retained rows · "
            f"{len(self.frame.columns)} source columns"
        )
        self.last_result = None
        self.export_result_button.setEnabled(False)
        self.result_table.clearContents()
        self.result_table.setRowCount(0)
        self.details.clear()
        self._refresh_preview()
        self._refresh_convention_evidence()

    def _refresh_preview(self) -> None:
        selected = tuple(item.text() for item in self.predictor_list.selectedItems())
        self.preview_panel.set_frame(
            self.frame,
            self.outcome_combo.currentText(),
            selected,
        )

    def _refresh_convention_evidence(self) -> None:
        convention = self.convention_combo.currentData()
        parameter_text = self.outcome_combo.currentText()
        try:
            parameter = ParameterId(parameter_text)
        except ValueError:
            parameter = ParameterId.CLUB_SPEED
        definition = convention_registry().definition(convention, parameter)
        reference = definition.reference_point.value.replace("_", " ")
        event_time = definition.event_time.value.replace("_", " ")
        self.convention_evidence.setText(
            f"<b>{definition.label}</b>: {reference}, {event_time}. "
            f"<a href='{definition.source_url}'>Source definition</a>"
        )

    def set_frame(
        self, frame: pd.DataFrame, source_name: str = "In-Memory Data"
    ) -> None:
        """Replace all records without discarding any source columns."""
        if len(frame) > MAX_RETAINED_ROWS:
            raise ValueError(f"The retained-data limit is {MAX_RETAINED_ROWS} rows")
        candidate = frame.copy()
        if len(candidate) < 3 or len(numeric_columns(candidate)) < 2:
            raise ValueError(
                "The dataset needs at least three rows and two numeric columns"
            )
        try:
            for column in candidate.columns:
                candidate[column].nunique(dropna=True)
        except TypeError as error:
            raise ValueError(
                "Launch-monitor records must contain flat scalar values"
            ) from error
        self.frame = candidate
        self.source_name = source_name
        self.preview_panel.reset_dataset()
        self._refresh_columns()

    def load_demo(self) -> None:
        self.set_frame(demo_frame(), "Built-In Demonstration Data")

    def import_path(self, path: Path) -> None:
        frame = read_launch_monitor_frame(path)
        if len(frame) < 3 or len(numeric_columns(frame)) < 2:
            raise ValueError(
                "The file needs at least three rows and two numeric columns"
            )
        self.set_frame(frame, path.name)

    def import_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Import Launch Monitor Data", "", "Data Files (*.csv *.json)"
        )
        if not selected:
            return
        try:
            self.import_path(Path(selected))
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Import Failed", str(error))

    def _selected_predictors(self) -> tuple[str, ...]:
        return tuple(item.text() for item in self.predictor_list.selectedItems())

    def run_analysis(self) -> AnalysisResult:
        group = self.group_combo.currentText()
        result = analyze_launch_monitor_data(
            self.frame,
            AnalysisRequest(
                outcome=self.outcome_combo.currentText(),
                predictors=self._selected_predictors(),
                analysis_mode=cast(AnalysisMode, self.mode_combo.currentText()),
                correlation_method=cast(
                    CorrelationMethod, self.method_combo.currentText()
                ),
                missing_policy=cast(MissingPolicy, self.missing_combo.currentText()),
                group_by=None if group == "(none)" else group,
                confidence_level=self.confidence_spin.value(),
                min_samples=self.min_samples_spin.value(),
            ),
        )
        render_analysis_result(result, self.result_table, self.details)
        self.last_result = result
        self.export_result_button.setEnabled(True)
        return result

    def run_analysis_safely(self) -> None:
        try:
            self.run_analysis()
        except ValueError as error:
            QMessageBox.warning(self, "Analysis Not Run", str(error))

    def export_data_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export Retained Data", "launch-monitor-records.json", "JSON (*.json)"
        )
        if selected:
            Path(selected).write_text(
                self.frame.to_json(orient="records", indent=2), encoding="utf-8"
            )

    def export_result_dialog(self) -> None:
        if self.last_result is None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Launch Monitor Analysis",
            "launch-monitor-analysis.json",
            "JSON (*.json)",
        )
        if selected:
            Path(selected).write_text(
                json.dumps(self.last_result.to_wire(), indent=2, sort_keys=True),
                encoding="utf-8",
            )


__all__ = ["LaunchMonitorAnalyticsTab"]
