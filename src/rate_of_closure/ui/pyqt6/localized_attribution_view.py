"""Accessible PyQt presentation for retained localized intervention pairs."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.variation.localized_attribution import (
    INTERPRETATION,
    AttributionAuthority,
    AttributionView,
    AttributionViewDefinition,
    attribution_observations_to_csv,
    attribution_view_to_json,
    build_attribution_view,
)

_CAVEAT = (
    "Paired planted-intervention response only; this view does not infer "
    "causality from Monte Carlo scatter or correlation."
)
_UNAVAILABLE = (
    "Attribution unavailable: this Monte Carlo result retains perturbed "
    "traces and scalar outcomes, but not isolated baseline/perturbed pairs. "
    "Scatter and rank correlation are not substituted for intervention authority."
)


class LocalizedAttributionView(QWidget):
    """Render strict source/target/pair authority or fail closed visibly."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._authority: AttributionAuthority | None = None
        self._caveat = QLabel(_CAVEAT)
        self._caveat.setWordWrap(True)
        self._caveat.setAccessibleName("Localized attribution interpretation")
        self._status = QLabel(_UNAVAILABLE)
        self._status.setWordWrap(True)
        self._status.setAccessibleName("Localized attribution availability")
        self._source = QComboBox()
        self._source.setAccessibleName("Localized attribution source specification")
        self._source.setToolTip(
            "Stable source spec ID with its topological joint and half-open window."
        )
        self._target = QComboBox()
        self._target.setAccessibleName("Localized attribution target")
        self._target.setToolTip(
            "State targets use spatial swing.* point/time; impact and shot targets "
            "retain typed unavailability. Target IDs are opaque stable selectors; "
            "the registry-owned name/unit/frame/convention defines meaning."
        )
        self._pair = QComboBox()
        self._pair.setAccessibleName("Localized attribution retained pair")
        self._pair.setToolTip("Explicit retained baseline and perturbed trial IDs.")
        self._locus = QLabel("—")
        self._locus.setWordWrap(True)
        self._locus.setAccessibleName("Localized attribution source and target locus")
        self._table = QTableWidget(1, 4)
        self._table.setHorizontalHeaderLabels(
            ("Baseline", "Perturbed", "Response", "Availability")
        )
        self._table.setAccessibleName("Localized attribution selected response")
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._denominator = QLabel("No paired authority loaded.")
        self._denominator.setAccessibleName("Localized attribution denominator")
        self._configure_run = QPushButton("Configure & Run Separate Paired Study…")
        self._configure_run.setAccessibleName("Configure separate paired study")
        self._configure_run.setToolTip(
            "Run a separate deterministic baseline/one-source intervention study. "
            "This does not reuse Monte Carlo scatter and adds 2 trials per "
            "localized source."
        )
        self._cancel_study = QPushButton("Cancel Paired Study")
        self._cancel_study.setAccessibleName("Cancel separate paired study")
        self._cancel_study.setEnabled(False)
        self._study_progress = QProgressBar()
        self._study_progress.setAccessibleName("Separate paired study progress")
        self._study_progress.setRange(0, 1)
        self._study_status = QLabel("No separate paired study is running.")
        self._study_status.setAccessibleName("Separate paired study status")
        self._study_status.setWordWrap(True)
        self._raw_export = QPushButton("Export Raw Observations CSV")
        self._raw_export.setAccessibleName("Export localized attribution observations")
        self._view_export = QPushButton("Export View Definition JSON")
        self._view_export.setAccessibleName(
            "Export localized attribution view definition"
        )
        self._save_authority = QPushButton("Save Paired Authority JSON…")
        self._save_authority.setAccessibleName("Save paired authority JSON")
        self._load_authority = QPushButton("Load Paired Authority JSON…")
        self._load_authority.setAccessibleName("Load archived paired authority JSON")
        self._build_layout()
        self._source.currentIndexChanged.connect(self._source_changed)
        self._target.currentIndexChanged.connect(self._target_changed)
        self._pair.currentIndexChanged.connect(self._refresh)
        self._raw_export.clicked.connect(self._export_raw)
        self._view_export.clicked.connect(self._export_view)
        self._set_controls_enabled(False)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._caveat)
        layout.addWidget(self._status)
        study_actions = QHBoxLayout()
        study_actions.addWidget(self._configure_run)
        study_actions.addWidget(self._cancel_study)
        study_actions.addStretch(1)
        layout.addLayout(study_actions)
        layout.addWidget(self._study_progress)
        layout.addWidget(self._study_status)
        controls = QFormLayout()
        controls.addRow("Source specification", self._source)
        controls.addRow("Target state / impact / shot", self._target)
        controls.addRow("Retained pair", self._pair)
        layout.addLayout(controls)
        layout.addWidget(self._locus)
        layout.addWidget(self._table)
        layout.addWidget(self._denominator)
        actions = QHBoxLayout()
        actions.addWidget(self._raw_export)
        actions.addWidget(self._view_export)
        actions.addWidget(self._save_authority)
        actions.addWidget(self._load_authority)
        actions.addStretch(1)
        layout.addLayout(actions)

    def set_authority(
        self, authority: AttributionAuthority | None, reason: str = _UNAVAILABLE
    ) -> None:
        """Replace authority atomically; ``None`` clears all prior evidence."""
        if authority is not None and not isinstance(authority, AttributionAuthority):
            raise TypeError("authority must be AttributionAuthority or None")
        self._authority = authority
        self._clear_combos()
        if authority is None:
            self._status.setText(reason)
            self._locus.setText("—")
            self._denominator.setText("No paired authority loaded.")
            self._clear_table()
            self._set_controls_enabled(False)
            return
        self._status.setText(
            f"Loaded {len(authority.pairs)} retained pairs · {INTERPRETATION}."
        )
        for source in authority.sources:
            self._source.addItem(source.spec_id, source.spec_id)
        self._set_controls_enabled(True)
        self._source_changed()

    def authority(self) -> AttributionAuthority | None:
        """Return the currently displayed immutable authority."""
        return self._authority

    def set_study_running(self, running: bool, total_runs: int = 1) -> None:
        """Update only the separate paired-study controls."""
        self._configure_run.setEnabled(not running)
        self._cancel_study.setEnabled(running)
        self._load_authority.setEnabled(not running)
        if running:
            self._study_progress.setRange(0, max(total_runs, 1))
            self._study_progress.setValue(0)

    def set_configure_enabled(self, enabled: bool, reason: str = "") -> None:
        """Expose paired-study capability without changing loaded authority."""
        if not self._cancel_study.isEnabled():
            self._configure_run.setEnabled(enabled)
        self._configure_run.setToolTip(
            reason
            or "Run a separate deterministic baseline/one-source intervention "
            "study. This does not reuse Monte Carlo scatter and adds 2 trials per "
            "localized source."
        )

    def set_study_status(self, text: str) -> None:
        """Set user-visible status for the independent paired-study lifecycle."""
        self._study_status.setText(text)

    def selected_view(self) -> AttributionView | None:
        """Resolve the current exact selection, if paired authority exists."""
        authority = self._authority
        raw_pair = self._pair.currentData()
        pair = tuple(raw_pair) if isinstance(raw_pair, (tuple, list)) else ()
        if authority is None or len(pair) != 2:
            return None
        definition = AttributionViewDefinition(
            authority.authority_id,
            str(self._source.currentData()),
            str(self._target.currentData()),
            int(pair[0]),
            int(pair[1]),
        )
        return build_attribution_view(authority, definition)

    def raw_csv(self) -> str | None:
        """Return strict raw observation CSV for the loaded authority."""
        return (
            None
            if self._authority is None
            else attribution_observations_to_csv(self._authority)
        )

    def view_json(self) -> str | None:
        """Return strict JSON persistence for the current view selection."""
        view = self.selected_view()
        if view is None or self._authority is None:
            return None
        pair = view.selected
        encoded: str = attribution_view_to_json(
            AttributionViewDefinition(
                self._authority.authority_id,
                view.source.spec_id,
                view.target.target_id,
                pair.baseline_trial_index,
                pair.perturbed_trial_index,
            )
        )
        return encoded

    def _clear_combos(self) -> None:
        for combo in (self._source, self._target, self._pair):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)

    def _source_changed(self) -> None:
        authority = self._authority
        source_id = self._source.currentData()
        self._target.blockSignals(True)
        self._target.clear()
        if authority is not None and isinstance(source_id, str):
            for target in authority.targets:
                self._target.addItem(f"{target.kind}: {target.name}", target.target_id)
        self._target.blockSignals(False)
        self._target_changed()

    def _target_changed(self) -> None:
        authority = self._authority
        source_id = self._source.currentData()
        self._pair.blockSignals(True)
        self._pair.clear()
        if authority is not None:
            for retained in authority.pairs:
                if retained.source_spec_id == source_id:
                    pair = (
                        retained.baseline_trial_index,
                        retained.perturbed_trial_index,
                    )
                    self._pair.addItem(f"Trial {pair[0]} → {pair[1]}", pair)
        self._pair.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        view = self.selected_view()
        if view is None:
            self._clear_table()
            return
        source, target, row = view.source, view.target, view.selected
        target_locus = (
            f"{target.point_id} at {target.time_s} s · {target.coordinate_frame}"
            if target.kind == "state"
            else f"{target.kind} outcome · {target.name}"
        )
        self._locus.setText(
            f"{source.joint_id} · [{source.time_window_s[0]}, "
            f"{source.time_window_s[1]}) s · {source.unit} → {target_locus} · "
            f"{target.convention} · opaque stable ID {target.target_id}"
        )
        values = (
            self._value(
                row.baseline_target_value, target.unit, row.baseline_status.value
            ),
            self._value(
                row.perturbed_target_value, target.unit, row.perturbed_status.value
            ),
            self._value(row.response, target.unit),
            row.availability.value,
        )
        for column, value in enumerate(values):
            self._table.setItem(0, column, QTableWidgetItem(value))
        denominator = view.denominator
        self._denominator.setText(
            f"Denominator: {denominator.available_pairs}/{denominator.total_pairs} "
            f"available · {denominator.typed_no_impact_pairs} typed no-impact · "
            f"{denominator.unavailable_no_impact_pairs} no-impact unavailable · "
            f"{denominator.failed_pairs} failed · "
            f"{denominator.nonfinite_pairs} nonfinite unavailable."
        )

    @staticmethod
    def _value(value: float | None, unit: str, status: str | None = None) -> str:
        text = "Unavailable" if value is None else f"{value:.7g} {unit}"
        return text if status is None else f"{text} · {status}"

    def _clear_table(self) -> None:
        for column in range(self._table.columnCount()):
            self._table.setItem(0, column, QTableWidgetItem("—"))

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._source,
            self._target,
            self._pair,
            self._raw_export,
            self._view_export,
            self._save_authority,
        ):
            widget.setEnabled(enabled)

    def _export_raw(self) -> None:
        self._write_dialog("localized_attribution_observations.csv", self.raw_csv())

    def _export_view(self) -> None:
        self._write_dialog("localized_attribution_view.json", self.view_json())

    def _write_dialog(self, suggested: str, content: str | None) -> None:
        if content is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export attribution", suggested)
        if path:
            Path(path).write_text(content, encoding="utf-8", newline="")


__all__ = ["LocalizedAttributionView"]
