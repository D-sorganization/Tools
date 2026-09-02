"""PyQt controls for hash-verified, course-state strokes gained."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TypeAlias

import pandas as pd
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from rate_of_closure.launch_monitor_strokes_gained import (
    SourceBackedStrokesGainedRequest,
    SourceBackedStrokesGainedResult,
    StrokesGainedBaseline,
    TrustedSummaryRequest,
    build_source_backed_strokes_gained_payload,
    calculate_source_backed_strokes_gained,
    load_strokes_gained_baseline,
)
from rate_of_closure.launch_monitor_v2_client import (
    StrokesGainedResponseV1,
    UpstreamV2Client,
)

ScoringResult: TypeAlias = SourceBackedStrokesGainedResult | StrokesGainedResponseV1


def _reason_text(result: SourceBackedStrokesGainedResult) -> str:
    """Render the exclusion audit trail for the status line (ADR-0048 G1-D3)."""

    by_reason = result.exclusions.by_reason
    if not by_reason:
        return "no exclusions"
    return ", ".join(f"{code} {count}" for code, count in sorted(by_reason.items()))


class LaunchMonitorSourceBackedStrokesGainedWidget(QWidget):
    """Load a verified baseline and map explicit before/after course state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._baseline: StrokesGainedBaseline | None = None
        self._baseline_path: Path | None = None
        self.result: ScoringResult | None = None
        self._build_ui()

    def _combo(self, name: str) -> QComboBox:
        combo = QComboBox()
        combo.setAccessibleName(name)
        combo.setToolTip(name)
        return combo

    def _build_ui(self) -> None:
        self.load_button = QPushButton("Load Verified Baseline...")
        self.authority_url = QLineEdit(os.getenv("UPSTREAMDRIFT_API_URL", ""))
        self.authority_url.setAccessibleName("Upstream strokes-gained authority URL")
        self.authority_url.setToolTip(
            "HTTP(S) UpstreamDrift API authority; blank uses the labeled local fallback"
        )
        self.before_lie = self._combo("Before lie column")
        self.before_context = self._combo("Before context column")
        self.before_target = self._combo("Before target or hole column")
        self.before_distance = self._combo("Before distance column")
        self.before_unit = self._combo("Before distance unit")
        self.after_lie = self._combo("After lie column")
        self.after_context = self._combo("After context column")
        self.after_target = self._combo("After target or hole column")
        self.after_distance = self._combo("After distance column")
        self.after_unit = self._combo("After distance unit")
        self.before_unit.addItems(["yd", "m"])
        self.after_unit.addItems(["yd", "m"])
        self.player_group = self._combo("Trusted player identity column for SG")
        self.session_group = self._combo("Trusted session identity column for SG")
        self.club_group = self._combo("Trusted club identity column for SG")
        self.order_column = self._combo("Explicit longitudinal order column for SG")
        self.summary_attest = QCheckBox(
            "Explicit trusted grouping identities and order"
        )
        self.summary_attest.setAccessibleName(
            "Attest strokes-gained grouping identities and longitudinal order"
        )
        self.summary_attest.setToolTip(
            "Enables canonical player, session, club, and longitudinal summaries; "
            "identity is never inferred"
        )
        self.calculate_button = QPushButton("Calculate Source-Backed SG")
        self.export_button = QPushButton("Export Source-Backed SG...")
        self.status = QLabel(
            "Unavailable until a versioned, hash-verified expected-strokes "
            "baseline artifact is loaded. No table is bundled."
        )
        self.status.setWordWrap(True)
        descriptions = (
            (self.load_button, "Load and SHA-256 verify a licensed baseline artifact"),
            (
                self.calculate_button,
                "Calculate verified E(before) minus one minus verified E(after)",
            ),
            (
                self.export_button,
                "Export baseline provenance, formula, lookups, and shot results",
            ),
        )
        for control, description in descriptions:
            control.setAccessibleName(description)
            control.setToolTip(description)
        self._install_layout()
        self.load_button.clicked.connect(self.load_dialog)
        self.calculate_button.clicked.connect(self.calculate_safely)
        self.export_button.clicked.connect(self.export_dialog)
        for combo in (
            self.before_lie,
            self.before_context,
            self.before_target,
            self.before_distance,
            self.after_lie,
            self.after_context,
            self.after_target,
            self.after_distance,
        ):
            combo.currentTextChanged.connect(self._refresh_enabled)
        self._refresh_enabled()

    def _install_layout(self) -> None:
        form = QFormLayout(self)
        form.addRow(self.load_button)
        form.addRow("Upstream authority URL:", self.authority_url)
        form.addRow("Before lie:", self.before_lie)
        form.addRow("Before context:", self.before_context)
        form.addRow("Before target/hole:", self.before_target)
        form.addRow("Before distance:", self.before_distance)
        form.addRow("Before unit:", self.before_unit)
        form.addRow("After lie:", self.after_lie)
        form.addRow("After context:", self.after_context)
        form.addRow("After target/hole:", self.after_target)
        form.addRow("After distance:", self.after_distance)
        form.addRow("After unit:", self.after_unit)
        form.addRow("Player summary:", self.player_group)
        form.addRow("Session summary:", self.session_group)
        form.addRow("Club summary:", self.club_group)
        form.addRow("Longitudinal order:", self.order_column)
        form.addRow(self.summary_attest)
        form.addRow(self.calculate_button)
        form.addRow(self.export_button)
        form.addRow(self.status)

    def set_dataset(self, frame: pd.DataFrame) -> None:
        """Bind retained rows while retaining an independently verified baseline."""

        self._frame = frame
        columns = sorted(str(column) for column in frame.columns)
        numeric = [
            str(column)
            for column in frame.columns
            if pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        for combo, values in (
            (self.before_lie, columns),
            (self.before_context, columns),
            (self.before_target, columns),
            (self.after_lie, columns),
            (self.after_context, columns),
            (self.after_target, columns),
            (self.player_group, columns),
            (self.session_group, columns),
            (self.club_group, columns),
            (self.before_distance, numeric),
            (self.after_distance, numeric),
            (self.order_column, numeric),
        ):
            combo.clear()
            combo.addItem("")
            combo.addItems(values)
        self.result = None
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        ready = self._baseline is not None and all(
            combo.currentText()
            for combo in (
                self.before_lie,
                self.before_context,
                self.before_target,
                self.before_distance,
                self.after_lie,
                self.after_context,
                self.after_target,
                self.after_distance,
            )
        )
        self.calculate_button.setEnabled(ready)
        self.export_button.setEnabled(self.result is not None)

    def load_path(self, path: Path) -> StrokesGainedBaseline:
        """Load, validate, and display one baseline artifact."""

        baseline = load_strokes_gained_baseline(path)
        self._baseline = baseline
        self._baseline_path = path
        self.result = None
        self.status.setText(
            f"Verified {baseline.baseline_id} · version {baseline.version} · "
            f"SHA-256 {baseline.table_sha256} · license {baseline.license}."
        )
        self._refresh_enabled()
        return baseline

    def load_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Load Strokes-Gained Baseline", "", "JSON (*.json)"
        )
        if selected:
            try:
                self.load_path(Path(selected))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                QMessageBox.warning(self, "Baseline Not Loaded", str(error))

    def _request(self) -> SourceBackedStrokesGainedRequest:
        summary = None
        if self.summary_attest.isChecked():
            summary = TrustedSummaryRequest(
                player_column=self.player_group.currentText(),
                session_column=self.session_group.currentText(),
                club_column=self.club_group.currentText(),
                order_column=self.order_column.currentText(),
            )
        return SourceBackedStrokesGainedRequest(
            self.before_lie.currentText(),
            self.before_context.currentText(),
            self.before_target.currentText(),
            self.before_distance.currentText(),
            self.after_lie.currentText(),
            self.after_context.currentText(),
            self.after_target.currentText(),
            self.after_distance.currentText(),
            self.before_unit.currentText(),
            self.after_unit.currentText(),
            summary,
        )

    def calculate(self) -> ScoringResult:
        if self._baseline is None:
            raise ValueError("a verified strokes-gained baseline is required")
        request = self._request()
        authority = self.authority_url.text().strip()
        if authority:
            canonical_result = UpstreamV2Client(authority).strokes_gained(
                build_source_backed_strokes_gained_payload(
                    self._frame, self._baseline, request
                )
            )
            if canonical_result.mean is None:
                raise ValueError("canonical strokes-gained estimate is unavailable")
            groups = canonical_result.payload.get("group_summaries", [])
            trends = canonical_result.payload.get("longitudinal_summaries", [])
            group_count = len(groups) if isinstance(groups, list) else 0
            trend_count = len(trends) if isinstance(trends, list) else 0
            message = (
                f"Canonical mean source-backed SG {canonical_result.mean:.3f} strokes "
                f"across {canonical_result.count} shots · "
                f"{group_count} group summaries · "
                f"{trend_count} longitudinal summaries · Upstream authority."
            )
            result: ScoringResult = canonical_result
        else:
            local_result = calculate_source_backed_strokes_gained(
                self._frame, self._baseline, request
            )
            if local_result.mean is None:
                raise ValueError(
                    "local compatibility strokes-gained estimate is unavailable: "
                    f"{local_result.exclusions.total_excluded} of "
                    f"{local_result.exclusions.input_row_count} rows were "
                    f"excluded ({_reason_text(local_result)})"
                )
            excluded = local_result.exclusions.total_excluded
            audit = (
                f" · {excluded} excluded ({_reason_text(local_result)})"
                if excluded
                else ""
            )
            message = (
                f"Local compatibility mean source-backed SG {local_result.mean:.3f} "
                f"strokes across {len(local_result.values)} shots"
                f"{audit} · status {local_result.status} · "
                f"{local_result.baseline_id} {local_result.baseline_version}."
            )
            result = local_result
        self.result = result
        self.status.setText(message)
        self._refresh_enabled()
        return result

    def calculate_safely(self) -> None:
        try:
            self.calculate()
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Source-Backed SG Unavailable", str(error))

    def document(self) -> dict[str, object] | None:
        if self.result is None:
            return None
        return {
            **asdict(self.result),
            "baseline_relative_path": str(self._baseline_path or ""),
        }

    def export_dialog(self) -> None:
        document = self.document()
        if document is None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export Source-Backed SG", "source-backed-sg.json", "JSON (*.json)"
        )
        if selected:
            Path(selected).write_text(json.dumps(document, indent=2), encoding="utf-8")


__all__ = ["LaunchMonitorSourceBackedStrokesGainedWidget"]
