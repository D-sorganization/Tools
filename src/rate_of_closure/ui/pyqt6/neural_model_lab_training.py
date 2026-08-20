"""Private training-request actions for the PyQt Neural Model Lab."""

from __future__ import annotations

import json
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from rate_of_closure.launch_monitor_import import read_launch_monitor_frame
from rate_of_closure.neural_lab_contract import (
    DatasetAuthority,
    TrainingSelection,
    build_training_manifest,
    load_capability_manifest,
)


class NeuralTrainingActionsMixin:
    """Reference-only training actions supplied to the lab widget."""

    def _show_capabilities(self, manifest: Any) -> None:
        self.capability.setPlainText(
            "\n\n".join(
                f"{item.vendor}: {item.state} — {item.row_count:,} rows / "
                f"{item.strict_row_count:,} strict\nArtifact: {item.artifact_state}\n"
                + "\n".join(f"• {reason}" for reason in item.blockers)
                for item in manifest.vendors
            )
        )

    def _load_capabilities(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Load private capability manifest", "", "JSON (*.json)"
        )
        if not name:
            return
        try:
            self._show_capabilities(load_capability_manifest(Path(name)))
            self.job_status.setPlainText(
                "Loaded user-authorized private capability metadata; its path "
                "and private rows were not persisted."
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Capabilities unavailable", str(error))

    def _load_dataset(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Select custom training data", "", "Data (*.csv *.json)"
        )
        if not name:
            return
        try:
            path = Path(name)
            frame = read_launch_monitor_frame(path)
            if frame.empty:
                raise ValueError("custom dataset is empty")
            self.frame = frame
            self.dataset_path = path
            self.dataset_sha = sha256(path.read_bytes()).hexdigest()
            self.dataset_status.setText(
                f"{path.name}: {len(frame):,} rows; SHA-256 {self.dataset_sha}\n"
                f"Columns: {', '.join(map(str, frame.columns))}"
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Dataset unavailable", str(error))

    def _manifest(self) -> dict[str, object]:
        if self.dataset_path is None:
            raise ValueError("select a custom dataset first")
        authority = DatasetAuthority(
            self.dataset_path.name,
            self.repository.text(),
            self.commit.text(),
            self.dataset_path.name,
            self.dataset_sha,
            len(self.frame),
        )
        selection = TrainingSelection(
            self.vendor.text(),
            tuple(
                value.strip()
                for value in self.features.text().split(",")
                if value.strip()
            ),
            tuple(
                value.strip()
                for value in self.targets.text().split(",")
                if value.strip()
            ),
            self.split_group.text(),
            self.approved.isChecked(),
        )
        return build_training_manifest(authority, self.frame, selection).to_wire()

    def _write_request(self, destination: Path) -> None:
        destination.write_text(
            json.dumps(self._manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _export_training(self) -> None:
        name, _ = QFileDialog.getSaveFileName(
            self,
            "Export training request",
            "neural-training-request.v2.json",
            "JSON (*.json)",
        )
        if not name:
            return
        try:
            self._write_request(Path(name))
            self.job_status.setPlainText(
                f"Exported reference-only request to {name}. No in-app training "
                "occurred."
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Request unavailable", str(error))

    def _submit_training(self) -> None:
        try:
            command = self.cli.text().strip()
            if not command:
                raise ValueError("configure the private training CLI executable")
            request_hash = sha256(
                json.dumps(self._manifest(), sort_keys=True).encode()
            ).hexdigest()[:12]
            temp = Path(tempfile.gettempdir()) / f"rate-neural-{request_hash}.json"
            self._write_request(temp)
            self._request_path = temp
            self.process.start(command, ["submit", "--request", str(temp)])
            self.job_status.setPlainText(
                "Submitting reference-only request to private CLI…"
            )
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Training unavailable", str(error))

    def _process_output(self) -> None:
        value = bytes(self.process.readAllStandardOutput()).decode(
            errors="replace"
        ) + bytes(self.process.readAllStandardError()).decode(errors="replace")
        if value:
            self.job_status.appendPlainText(value.rstrip())

    def _monitor(self) -> None:
        states = {
            QProcess.ProcessState.NotRunning: "not running",
            QProcess.ProcessState.Starting: "starting",
            QProcess.ProcessState.Running: "running",
        }
        self.job_status.appendPlainText(
            f"Private CLI is {states[self.process.state()]}; "
            f"PID {self.process.processId()}."
        )


__all__ = ["NeuralTrainingActionsMixin"]
