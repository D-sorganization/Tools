"""Atomic save and export commands for the regional-ground execution workspace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from PyQt6.QtWidgets import QFileDialog, QWidget

from rate_of_closure.application.regional_ground_execution_files import (
    write_regional_ground_execution_job_atomic,
    write_regional_ground_execution_result_atomic,
    write_regional_ground_execution_rows_csv_atomic,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)


class RegionalGroundExecutionFilesMixin:
    """Save the job, save the result, and export rows, always atomically.

    Split out of the workspace purely for the 500 LOC budget. These stay
    methods rather than free functions because each one parents a modal
    ``QFileDialog`` on the workspace and reports through its status label.
    """

    # Supplied by the concrete workspace; declared so this module type-checks
    # on its own under CI's ``--follow-imports=skip``.
    _job: RegionalGroundExecutionJob | None
    _result: RegionalGroundExecutionResult | None
    _recent_path: Path | None

    _set_status: Callable[[str, str], None]

    def save_job_as(self) -> None:
        job = self._job
        if job is None:
            self._set_status("Save unavailable: no execution job loaded.", "error")
            return
        self._save(
            "Save Regional-Ground Execution Job As",
            "regional-ground-execution-job.json",
            lambda path: write_regional_ground_execution_job_atomic(job, path),
        )

    def save_result_as(self) -> None:
        result = self._result
        if result is None:
            self._set_status("Save unavailable: no complete result retained.", "error")
            return
        self._save(
            "Save Regional-Ground Execution Result As",
            "regional-ground-execution-result.json",
            lambda path: write_regional_ground_execution_result_atomic(result, path),
        )

    def export_rows_csv(self) -> None:
        result = self._result
        if result is None:
            self._set_status(
                "Export unavailable: no complete result retained.", "error"
            )
            return
        self._save(
            "Export Regional-Ground Execution Rows",
            "regional-ground-execution-rows.csv",
            lambda path: write_regional_ground_execution_rows_csv_atomic(result, path),
            file_filter="CSV files (*.csv)",
        )

    def _save(
        self,
        title: str,
        filename: str,
        writer: Callable[[Path], bool],
        *,
        file_filter: str = "JSON files (*.json)",
    ) -> None:
        # This mixin is only ever mixed into a QWidget, so the dialog can be
        # parented on self; the cast states that host requirement to mypy.
        selected, _filter = QFileDialog.getSaveFileName(
            cast(QWidget, self), title, self._initial_location(filename), file_filter
        )
        if not selected:
            return
        path = Path(selected)
        try:
            writer(path)
        except (OSError, TypeError, ValueError):
            self._set_status(
                "Save failed: the destination could not be written atomically. "
                "Retained evidence was preserved.",
                "error",
            )
            return
        self._recent_path = path
        self._set_status(f"Saved {path.name} atomically.", "success")

    def _initial_location(self, filename: str = "") -> str:
        if self._recent_path is None:
            return filename
        return (
            str(self._recent_path.parent / filename)
            if filename
            else str(self._recent_path.parent)
        )


__all__ = ["RegionalGroundExecutionFilesMixin"]
