"""PyQt Neural Model Lab: private training client and safe JSON inference."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.neural_lab_contract import (
    PortableModel,
    load_capability_manifest,
    load_portable_model,
    predict_one,
)
from rate_of_closure.ui.pyqt6.neural_model_lab_training import (
    NeuralTrainingActionsMixin,
)
from rate_of_closure.ui.pyqt6.neural_model_lab_widgets import (
    CapabilityCanvas,
    ResidualPlot,
)


class NeuralModelLabTab(NeuralTrainingActionsMixin, QWidget):
    """Fail-closed client surface; training executes only in a private authority."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frame = pd.DataFrame()
        self.dataset_path: Path | None = None
        self.dataset_sha = ""
        self.model: PortableModel | None = None
        self.model_payload: dict[str, object] | None = None
        self.query_fields: dict[str, QLineEdit] = {}
        self._request_path: Path | None = None
        self.process = QProcess(self)
        self._build_ui()
        self._connect()

    @staticmethod
    def _line(label: str) -> QLineEdit:
        field = QLineEdit()
        field.setAccessibleName(label)
        field.setToolTip(label)
        return field

    def _build_ui(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        heading = QLabel("Neural Model Lab")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        boundary = QLabel(
            "Safe client for private, group-safe vendor-comparable surrogate "
            "training. This app never trains in-process. Portable JSON inference "
            "is descriptive, not device emulation or certification."
        )
        boundary.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(boundary)
        manifest = load_capability_manifest()
        self.capability = QPlainTextEdit()
        self.capability.setReadOnly(True)
        self.capability.setAccessibleName("Vendor Neural Capability Manifest")
        self.capability.setToolTip(
            "Availability is read from the versioned capability manifest, never "
            "hard-coded in this UI."
        )
        self._show_capabilities(manifest)
        self.capability_plot = CapabilityCanvas()
        self.capability_plot.set_capabilities(manifest.vendors)
        self.load_capabilities = QPushButton("Load Capability Manifest…")
        self.load_capabilities.setAccessibleName(
            "Load a user-authorized private capability manifest"
        )
        self.load_capabilities.setToolTip(
            "Validates private capability metadata without persisting its path "
            "or any private rows."
        )
        layout.addWidget(self.load_capabilities)
        layout.addWidget(self.capability_plot)
        layout.addWidget(self.capability)

        self.dataset_button = QPushButton("Select Custom Dataset…")
        self.dataset_status = QLabel("No custom dataset loaded.")
        self.dataset_status.setWordWrap(True)
        self.repository = self._line("Private dataset repository")
        self.commit = self._line("Immutable 40-character dataset commit")
        self.vendor = self._line("Explicit vendor or custom model family")
        self.vendor.setText("Custom")
        self.features = self._line("Feature columns, comma separated")
        self.targets = self._line("Target columns, comma separated")
        self.split_group = self._line("Policy-approved repeating split group column")
        self.approved = QCheckBox(
            "I attest this is a policy-approved repeating split group"
        )
        self.approved.setAccessibleName("Policy-approved repeating split group")
        self.approved.setToolTip(
            "Must represent at least three independent groups with at least one "
            "repeated group; row IDs are forbidden."
        )
        self.cli = self._line("Private training CLI executable")
        self.export_request = QPushButton("Export Training Request…")
        self.submit_request = QPushButton("Submit to Private CLI")
        self.monitor = QPushButton("Monitor Private Job")
        for button, tip in (
            (
                self.dataset_button,
                "Select CSV or JSON custom data and compute its immutable SHA-256",
            ),
            (
                self.export_request,
                "Validate and export a reference-only JSON training request",
            ),
            (
                self.submit_request,
                "Invoke the configured private CLI with the reference-only request",
            ),
            (self.monitor, "Show private CLI state and captured non-row output"),
        ):
            button.setAccessibleName(tip)
            button.setToolTip(tip)
        form = QFormLayout()
        form.addRow(self.dataset_button)
        form.addRow(self.dataset_status)
        for label, widget in (
            ("Repository:", self.repository),
            ("Commit:", self.commit),
            ("Vendor:", self.vendor),
            ("Features:", self.features),
            ("Targets:", self.targets),
            ("Split Group:", self.split_group),
            ("Private CLI:", self.cli),
        ):
            form.addRow(label, widget)
        form.addRow(self.approved)
        actions = QHBoxLayout()
        actions.addWidget(self.export_request)
        actions.addWidget(self.submit_request)
        actions.addWidget(self.monitor)
        form.addRow(actions)
        layout.addLayout(form)
        self.job_status = QPlainTextEdit()
        self.job_status.setReadOnly(True)
        self.job_status.setAccessibleName("Private Training Job Status")
        self.job_status.setPlainText("No private training request submitted.")
        layout.addWidget(self.job_status)

        self.load_model = QPushButton("Load Portable Model…")
        self.export_model = QPushButton("Export Model Inspection…")
        self.query_button = QPushButton("Query Model")
        self.query_button.setEnabled(False)
        self.export_model.setEnabled(False)
        for button, tip in (
            (
                self.load_model,
                "Load bounded non-executable JSON after schema and provenance "
                "hash validation",
            ),
            (
                self.query_button,
                "Run local deterministic inference and report out-of-domain warnings",
            ),
            (
                self.export_model,
                "Export the validated model card, metrics, residual evidence, "
                "and provenance",
            ),
        ):
            button.setAccessibleName(tip)
            button.setToolTip(tip)
        model_actions = QHBoxLayout()
        model_actions.addWidget(self.load_model)
        model_actions.addWidget(self.query_button)
        model_actions.addWidget(self.export_model)
        layout.addLayout(model_actions)
        self.model_summary = QPlainTextEdit()
        self.model_summary.setReadOnly(True)
        self.model_summary.setAccessibleName(
            "Validated Model Card Metrics and Prediction"
        )
        self.query_form = QFormLayout()
        layout.addLayout(self.query_form)
        layout.addWidget(self.model_summary)
        self.residual_plot = ResidualPlot()
        layout.addWidget(self.residual_plot)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.addWidget(scroll)

    def _connect(self) -> None:
        self.load_capabilities.clicked.connect(self._load_capabilities)
        self.dataset_button.clicked.connect(self._load_dataset)
        self.export_request.clicked.connect(self._export_training)
        self.submit_request.clicked.connect(self._submit_training)
        self.monitor.clicked.connect(self._monitor)
        self.load_model.clicked.connect(self._load_model)
        self.query_button.clicked.connect(self._query)
        self.export_model.clicked.connect(self._export_model)
        self.process.readyReadStandardOutput.connect(self._process_output)
        self.process.readyReadStandardError.connect(self._process_output)
        self.process.finished.connect(
            lambda code: self.job_status.appendPlainText(
                f"Private CLI finished with exit code {code}."
            )
        )

    def _load_model(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Load portable neural model", "", "JSON (*.json)"
        )
        if not name:
            return
        try:
            payload = json.loads(Path(name).read_text(encoding="utf-8"))
            model = load_portable_model(payload)
            self.model, self.model_payload = model, payload
            while self.query_form.rowCount():
                self.query_form.removeRow(0)
            self.query_fields = {}
            for feature in model.features:
                field = self._line(
                    f"{feature.name} query in {feature.unit}; training range "
                    f"{feature.minimum} to {feature.maximum}"
                )
                field.setText(str(feature.mean))
                self.query_fields[feature.name] = field
                self.query_form.addRow(f"{feature.name} ({feature.unit}):", field)
            self.model_summary.setPlainText(
                json.dumps(
                    {
                        "model_id": model.model_id,
                        "vendor": model.vendor,
                        "model_card": model.model_card,
                        "metrics": model.metrics,
                    },
                    indent=2,
                )
            )
            self.residual_plot.set_residuals(dict(model.residuals))
            self.query_button.setEnabled(True)
            self.export_model.setEnabled(True)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Model unavailable", str(error))

    def _query(self) -> None:
        if self.model is None:
            return
        try:
            result = predict_one(
                self.model,
                {
                    name: float(field.text())
                    for name, field in self.query_fields.items()
                },
            )
            self.model_summary.appendPlainText(
                "\nPrediction:\n"
                + json.dumps(
                    {"values": result.values, "warnings": result.warnings}, indent=2
                )
            )
        except ValueError as error:
            QMessageBox.warning(self, "Query unavailable", str(error))

    def _export_model(self) -> None:
        if self.model_payload is None:
            return
        name, _ = QFileDialog.getSaveFileName(
            self,
            "Export model inspection",
            "neural-model-inspection.json",
            "JSON (*.json)",
        )
        if name:
            Path(name).write_text(
                json.dumps(self.model_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
