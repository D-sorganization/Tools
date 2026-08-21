"""Reference-only canonical authority actions for the PyQt player workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from rate_of_closure.launch_monitor_v2_client import (
    CanonicalDatasetReference,
    UpstreamV2Client,
    build_dataset_job_request,
    build_player_covariation_payload,
    load_canonical_dataset_reference,
)
from rate_of_closure.launch_monitor_workspace import LaunchMonitorProject


class CanonicalWorkspaceMixin:
    """Keep private-reference transport separate from local statistics/UI layout."""

    authority_url: QLineEdit
    status: QLabel
    corpus_reference_button: QPushButton
    inspect_corpus_button: QPushButton
    refresh_corpus_button: QPushButton
    canonical_covariation_button: QPushButton
    canonical_limit: QLabel
    _frame: pd.DataFrame
    _canonical_reference: CanonicalDatasetReference | None
    _canonical_job_id: str | None
    _export_payload: dict[str, object]

    def project(self) -> LaunchMonitorProject:
        """Return the host workspace project."""

        raise NotImplementedError

    def _refresh_enabled(self) -> None:
        """Refresh host controls after authority state changes."""

        raise NotImplementedError

    def _host_widget(self) -> QWidget:
        if not isinstance(self, QWidget):
            raise TypeError("canonical workspace mixin requires a QWidget host")
        return self

    def _build_canonical_controls(self) -> None:
        """Create the shared canonical authority controls for the host layout."""

        self.authority_url = QLineEdit()
        self.authority_url.setPlaceholderText("https://authorized-upstream.example")
        self.corpus_reference_button = QPushButton(
            "Load Authorized Corpus Reference..."
        )
        self.inspect_corpus_button = QPushButton("Inspect Authorized Corpus")
        self.refresh_corpus_button = QPushButton("Refresh Corpus Job")
        self.canonical_covariation_button = QPushButton(
            "Run Canonical Player Covariation"
        )
        self.canonical_limit = QLabel(
            "Canonical inline limit is 20,000 rows. Larger authorized corpora "
            "use reference-only aggregate jobs; private rows are never persisted."
        )
        self.canonical_limit.setWordWrap(True)

    def load_corpus_reference_dialog(self) -> None:
        """Load an immutable reference selected by the user, never corpus rows."""

        selected, _ = QFileDialog.getOpenFileName(
            self._host_widget(),
            "Load Authorized Corpus Reference",
            "",
            "JSON (*.json)",
        )
        if not selected:
            return
        try:
            payload = json.loads(Path(selected).read_text(encoding="utf-8"))
            self._canonical_reference = load_canonical_dataset_reference(payload)
            row_count = self._canonical_reference.expected_row_count
            self.status.setText(
                f"Authorized reference loaded: {row_count:,} rows; no rows were loaded."
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self._host_widget(), "Corpus Reference Not Loaded", str(error)
            )
        self._refresh_enabled()

    def submit_corpus_job_safely(self) -> None:
        """Submit a reference-only source summary to the configured authority."""

        try:
            if self._canonical_reference is None:
                raise ValueError("Load an authorized corpus reference first")
            client = UpstreamV2Client(self.authority_url.text())
            response = client.submit_dataset_job(
                build_dataset_job_request(self._canonical_reference, "source_summary")
            )
            self._canonical_job_id = str(response["job_id"])
            self.status.setText(
                f"Canonical corpus job {self._canonical_job_id} is "
                f"{response['status']}."
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self._host_widget(), "Corpus Job Not Submitted", str(error)
            )
        self._refresh_enabled()

    def refresh_corpus_job_safely(self) -> None:
        """Fetch bounded aggregate results only after canonical completion."""

        try:
            if self._canonical_job_id is None:
                raise ValueError("No canonical corpus job has been submitted")
            client = UpstreamV2Client(self.authority_url.text())
            response = client.dataset_job_status(self._canonical_job_id)
            if response["status"] == "completed":
                page = client.dataset_job_results(self._canonical_job_id)
                self._export_payload = page
                self.status.setText(
                    "Canonical corpus job completed with "
                    f"{page['total_items']} bounded aggregate items."
                )
            else:
                self.status.setText(f"Canonical corpus job is {response['status']}.")
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self._host_widget(), "Corpus Job Not Refreshed", str(error)
            )
        self._refresh_enabled()

    def run_canonical_covariation_safely(self) -> None:
        """Submit inline rows only within the canonical 20,000-row boundary."""

        try:
            project = self.project()
            records: list[dict[str, Any]] = [
                {str(key): value for key, value in row.items()}
                for row in self._frame.to_dict(orient="records")
            ]
            payload = build_player_covariation_payload(
                records,
                player_column=project.identity.column,
                x_column=project.selection.x,
                y_column=project.selection.y,
                min_samples=max(4, project.selection.min_samples),
                confidence_level=project.selection.confidence_level,
            )
            client = UpstreamV2Client(self.authority_url.text())
            self._export_payload = client.player_covariation(payload)
            self.status.setText(
                "Canonical Upstream player covariation completed with "
                "evidence-bearing lineage."
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self._host_widget(), "Canonical Covariation Not Run", str(error)
            )
        self._refresh_enabled()


__all__ = ["CanonicalWorkspaceMixin"]
