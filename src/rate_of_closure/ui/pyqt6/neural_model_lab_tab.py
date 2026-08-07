"""Interactive PyQt6 laboratory for private neural-surrogate training and query."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pandas as pd
from PyQt6.QtCore import QProcess, QProcessEnvironment
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.launch_monitor_analysis import numeric_columns
from rate_of_closure.launch_monitor_data import (
    CampaignDatasetCatalog,
    campaign_dataset_catalog,
    discover_campaign_repository,
    load_campaign_dataset,
    load_imported_dataset,
)
from rate_of_closure.neural_training import (
    TrainingOptions,
    TrainingRequest,
    build_training_request,
    discover_training_repository,
    parse_hidden_layers,
    write_training_config,
)
from rate_of_closure.ui.pyqt6.neural_model_outputs import NeuralModelOutputs
from rate_of_closure.ui.pyqt6.neural_training_controls import NeuralTrainingControls


class NeuralModelLabTab(QWidget):
    """Configure private training, inspect safe bundles, and query predictions."""

    def __init__(
        self, parent: QWidget | None = None, *, auto_discover_campaign: bool = True
    ) -> None:
        super().__init__(parent)
        self.frame = pd.DataFrame()
        self.dataset_path: Path | None = None
        self.campaign_root: Path | None = None
        self.catalog: CampaignDatasetCatalog | None = None
        self.process: QProcess | None = None
        self._build_ui()
        if auto_discover_campaign:
            self.refresh_campaign()
        else:
            self._refresh_columns()

    @staticmethod
    def _help(widget: QWidget, name: str, tip: str) -> None:
        widget.setAccessibleName(name)
        widget.setToolTip(tip)

    def _build_ui(self) -> None:
        heading = QLabel("Neural Model Lab")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        boundary = QLabel(
            "Train models only through the private campaign. Imported bundles are "
            "validated, non-executable JSON surrogates; vendor-comparable does not "
            "mean vendor-certified or reveal a proprietary algorithm."
        )
        boundary.setWordWrap(True)
        self.source_label = QLabel("No dataset loaded")
        self.source_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(boundary)
        layout.addLayout(self._data_toolbar())
        layout.addWidget(self.source_label)
        splitter = QSplitter()
        self.controls = NeuralTrainingControls()
        self.controls.train_button.clicked.connect(self._train_dialog)
        self.controls.cancel_button.clicked.connect(self.cancel_training)
        splitter.addWidget(self.controls)
        self.outputs: NeuralModelOutputs = NeuralModelOutputs()
        splitter.addWidget(self.outputs)
        splitter.setSizes([430, 980])
        layout.addWidget(splitter, 1)

    def _data_toolbar(self) -> QHBoxLayout:
        self.dataset_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh Campaign")
        self.import_data_button = QPushButton("Import Custom Data...")
        self.import_model_button = QPushButton("Import Model...")
        row = QHBoxLayout()
        for widget in (
            self.dataset_combo,
            self.refresh_button,
            self.import_data_button,
            self.import_model_button,
        ):
            row.addWidget(widget)
        self._help(
            self.dataset_combo,
            "Training Dataset",
            "Select a full manifested campaign table",
        )
        self._help(
            self.refresh_button,
            "Refresh Campaign",
            "Rediscover private datasets and model outputs",
        )
        self._help(
            self.import_data_button,
            "Import Custom Dataset",
            "Load a local CSV or record-array JSON",
        )
        self._help(
            self.import_model_button,
            "Import Neural Model",
            "Load a safe launch-monitor-neural-bundle/v1 JSON file",
        )
        self.dataset_combo.currentIndexChanged.connect(self._dataset_selected)
        self.refresh_button.clicked.connect(self.refresh_campaign)
        self.import_data_button.clicked.connect(self._import_data_dialog)
        self.import_model_button.clicked.connect(self._import_model_dialog)
        return row

    def interactive_controls(self) -> tuple[QWidget, ...]:
        """Return controls covered by the accessible-help contract."""

        return (
            self.dataset_combo,
            self.refresh_button,
            self.import_data_button,
            self.import_model_button,
            *self.controls.interactive_controls(),
            self.outputs.query_table,
            self.outputs.query_button,
            self.outputs.batch_button,
            self.outputs.export_predictions_button,
        )

    def refresh_campaign(self) -> None:
        """Discover private tables and load the normalized shot dataset."""

        root = discover_campaign_repository()
        if root is None:
            self.source_label.setText(
                "Private campaign not found; import a custom dataset to continue."
            )
            return
        self.campaign_root = discover_training_repository(root)
        self.catalog = campaign_dataset_catalog(root)
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for descriptor in self.catalog.datasets:
            self.dataset_combo.addItem(
                f"{descriptor.label} — {descriptor.row_count:,} rows",
                descriptor.dataset_id,
            )
        self.dataset_combo.blockSignals(False)
        normalized = self.dataset_combo.findData("normalized")
        self.dataset_combo.setCurrentIndex(max(0, normalized))
        self._dataset_selected(self.dataset_combo.currentIndex())

    def _dataset_selected(self, index: int) -> None:
        if self.catalog is None or index < 0:
            return
        dataset_id = self.dataset_combo.itemData(index)
        descriptor = next(
            item for item in self.catalog.datasets if item.dataset_id == dataset_id
        )
        self.dataset_path = descriptor.path
        self.set_dataset(
            load_campaign_dataset(descriptor), source_name=descriptor.label
        )

    def set_dataset(self, frame: pd.DataFrame, *, source_name: str) -> None:
        """Install a complete dataframe and refresh model-variable choices."""

        self.frame = frame.copy()
        self.outputs.set_frame(frame)
        self.source_label.setText(
            f"{source_name}: {len(frame):,} rows × {len(frame.columns):,} columns"
        )
        self._refresh_columns()

    def _refresh_columns(self) -> None:
        columns = numeric_columns(self.frame)
        self.controls.feature_list.clear()
        self.controls.target_list.clear()
        self.controls.split_group_combo.clear()
        self.controls.split_group_combo.addItems(
            [str(column) for column in self.frame.columns]
        )
        preferred = self.controls.split_group_combo.findText("shot_id")
        if preferred >= 0:
            self.controls.split_group_combo.setCurrentIndex(preferred)
        for column in columns:
            self.controls.feature_list.addItem(QListWidgetItem(column))
            self.controls.target_list.addItem(QListWidgetItem(column))
        self._enable_trackman_if_supported()

    def _enable_trackman_if_supported(self) -> None:
        combo = self.controls.vendor_combo
        index = combo.findText("TrackMan-comparable")
        model = combo.model()
        assert isinstance(model, QStandardItemModel)
        has_targets = any(
            name in self.frame for name in ("observed_carry_m", "carry_yd", "carry_m")
        )
        item = model.item(index)
        assert item is not None
        item.setEnabled(has_targets)
        reason = (
            "Available for the selected shot-level dataset."
            if has_targets
            else "Requires shot-level TrackMan outcomes in the selected dataset."
        )
        combo.setItemData(index, reason, 3)
        if has_targets:
            combo.setCurrentIndex(index)

    def _selected(self, widget: QListWidget) -> tuple[str, ...]:
        return tuple(item.text() for item in widget.selectedItems())

    def training_request(self, output_path: Path) -> TrainingRequest:
        """Persist current options and return the exact private CLI request."""

        if self.campaign_root is None or self.dataset_path is None:
            raise ValueError(
                "select a campaign or custom dataset and private campaign root"
            )
        options = TrainingOptions(
            vendor=str(self.controls.vendor_combo.currentData()),
            features=self._selected(self.controls.feature_list),
            targets=self._selected(self.controls.target_list),
            hidden_layers=parse_hidden_layers(self.controls.hidden_layers_edit.text()),
            activation=self.controls.activation_combo.currentText(),
            alpha=self.controls.alpha_spin.value(),
            seed=self.controls.seed_spin.value(),
            epochs=self.controls.epochs_spin.value(),
            holdout=self.controls.holdout_spin.value(),
            split_group=self.controls.split_group_combo.currentText(),
        )
        config_path = output_path.with_suffix(".training.toml")
        write_training_config(
            config_path,
            dataset_path=self.dataset_path,
            output_path=output_path,
            options=options,
        )
        return build_training_request(self.campaign_root, config_path)

    def start_training(self, output_path: Path) -> None:
        """Launch private training asynchronously and stream its output."""

        request = self.training_request(output_path)
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        inherited = environment.value("PYTHONPATH")
        environment.insert(
            "PYTHONPATH",
            os.pathsep.join(filter(None, (request.python_path, inherited))),
        )
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(request.working_directory))
        process.setProgram(request.program)
        process.setArguments(list(request.arguments))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._training_finished)
        self.process = process
        self.controls.train_button.setEnabled(False)
        self.controls.cancel_button.setEnabled(True)
        self.controls.progress.setRange(0, 0)
        self.outputs.log.appendPlainText(
            "Command: " + " ".join((request.program, *request.arguments))
        )
        process.start()

    def cancel_training(self) -> None:
        """Request termination of an active training subprocess."""

        if (
            self.process is not None
            and self.process.state() != QProcess.ProcessState.NotRunning
        ):
            self.process.terminate()
            self.outputs.log.appendPlainText("Cancellation requested.")

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        text = (
            bytes(self.process.readAllStandardOutput().data())
            .decode("utf-8", errors="replace")
            .rstrip()
        )
        if text:
            self.outputs.log.appendPlainText(text)

    def _training_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self.controls.progress.setRange(0, 100)
        self.controls.progress.setValue(100 if exit_code == 0 else 0)
        self.controls.train_button.setEnabled(True)
        self.controls.cancel_button.setEnabled(False)
        self.outputs.log.appendPlainText(
            f"Training process exited with code {exit_code}."
        )

    def import_model(self, path: Path) -> None:
        """Load and display a safe JSON bundle."""

        self.outputs.import_model(path)

    def predict_current_dataset(self) -> pd.DataFrame:
        """Evaluate every retained row of the current dataset."""

        output: pd.DataFrame = self.outputs.predict_current_dataset()
        return output

    def _import_data_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Import Custom Training Data", "", "Data (*.csv *.json)"
        )
        if selected:
            self.dataset_path = Path(selected).resolve()
            self.set_dataset(
                load_imported_dataset(self.dataset_path),
                source_name=self.dataset_path.name,
            )

    def _import_model_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Import Neural Bundle", "", "Neural JSON (*.json)"
        )
        if selected:
            self.import_model(Path(selected))

    def _train_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self, "Save Trained Neural Bundle", "model.nn.json", "Neural JSON (*.json)"
        )
        if selected:
            try:
                self.start_training(Path(selected))
            except ValueError as exc:
                QMessageBox.warning(self, "Training Configuration", str(exc))

    @property
    def model_summary(self) -> QPlainTextEdit:
        """Expose the summary widget for presentation contracts."""

        return cast(QPlainTextEdit, self.outputs.model_summary)

    @property
    def export_predictions_button(self) -> QPushButton:
        """Expose the export control for presentation contracts."""

        return cast(QPushButton, self.outputs.export_predictions_button)


__all__ = ["NeuralModelLabTab"]
