from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from rate_of_closure.launch_monitor_analysis import numeric_columns
from rate_of_closure.launch_monitor_data import (
    AnalysisProject,
    CampaignDatasetCatalog,
    DatasetDescriptor,
    campaign_dataset_catalog,
    discover_campaign_repository,
    file_sha256,
    load_analysis_project,
    load_campaign_dataset,
    load_imported_dataset,
    resolve_project_data,
    save_analysis_project,
)
from rate_of_closure.ui.pyqt6 import (
    launch_monitor_covariation_state as covariance_state,
)
from rate_of_closure.ui.pyqt6.launch_monitor_demo_data import demo_frame
from rate_of_closure.ui.pyqt6.launch_monitor_table_preview import (
    populate_table_preview,
)


class LaunchMonitorDataMixin:
    catalog: CampaignDatasetCatalog | None
    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...

    def refresh_campaign_catalog(self, explicit: Path | None = None) -> None:
        root = discover_campaign_repository(explicit)
        if root is None:
            self.source_label.setText(
                "Private campaign repository not found; demo remains loaded."
            )
            return
        try:
            self.catalog = campaign_dataset_catalog(root)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.warning(
                cast(QWidget, self), "Campaign Catalog Failed", str(error)
            )
            return
        self.dataset_combo.blockSignals(True)
        self._populate_catalog_combo(self.catalog)
        preferred = self.dataset_combo.findData("normalized")
        self.dataset_combo.setCurrentIndex(preferred if preferred >= 0 else 1)
        self.dataset_combo.blockSignals(False)
        self._dataset_selected()

    def _populate_catalog_combo(self, catalog: CampaignDatasetCatalog) -> None:
        self.dataset_combo.clear()
        self.dataset_combo.addItem("Built-In Demonstration Data", "demo")
        for descriptor in catalog.datasets:
            label = (
                f"{descriptor.label} — {descriptor.row_count:,} × "
                f"{descriptor.column_count}"
            )
            self.dataset_combo.addItem(label, descriptor.dataset_id)

    def _descriptor(self, dataset_id: str) -> DatasetDescriptor | None:
        if self.catalog is None:
            return None
        return next(
            (item for item in self.catalog.datasets if item.dataset_id == dataset_id),
            None,
        )

    def _dataset_selected(self) -> None:
        dataset_id = str(self.dataset_combo.currentData() or "demo")
        if dataset_id == "demo":
            self.load_demo()
            return
        descriptor = self._descriptor(dataset_id)
        if descriptor is None:
            return
        try:
            frame = load_campaign_dataset(descriptor)
        except (OSError, ValueError) as error:
            QMessageBox.critical(cast(QWidget, self), "Dataset Load Failed", str(error))
            return
        self.set_frame(
            frame,
            descriptor.label,
            dataset_id,
            descriptor.sha256,
            descriptor.path,
        )

    def set_frame(
        self,
        frame: pd.DataFrame,
        source_name: str = "In-Memory Data",
        dataset_id: str = "memory",
        source_sha256: str = "",
        data_path: Path | None = None,
    ) -> None:
        self.frame = frame.copy()
        self.source_name = source_name
        self.dataset_id = dataset_id
        self.source_sha256 = source_sha256
        self.data_path = str(data_path.resolve()) if data_path else ""
        self._refresh_columns()

    def load_demo(self) -> None:
        self.set_frame(demo_frame(), "Built-In Demonstration Data", "demo")

    def _refresh_columns(self) -> None:
        numeric = numeric_columns(self.frame)
        self.outcome_combo.clear()
        self.outcome_combo.addItems(numeric)
        preferred_outcome = next(
            (
                name
                for name in ("observed_carry_m", "ball_speed", "ball_speed_mph")
                if name in numeric
            ),
            numeric[0] if numeric else "",
        )
        self.outcome_combo.setCurrentText(preferred_outcome)
        self.predictor_list.clear()
        self.predictor_list.addItems(numeric)
        preferred_predictors = {
            name
            for name in (
                "ball_speed_mph",
                "club_speed",
                "launch_angle_deg",
                "attack_angle",
            )
            if name != preferred_outcome
        }
        for index in range(self.predictor_list.count()):
            item = self.predictor_list.item(index)
            if item is not None:
                item.setSelected(item.text() in preferred_predictors)
        if not self.predictor_list.selectedItems() and len(numeric) > 1:
            selected_index = 1 if numeric[0] == preferred_outcome else 0
            item = self.predictor_list.item(selected_index)
            if item is not None:
                item.setSelected(True)
        groups = sorted(
            str(column)
            for column in self.frame.columns
            if self.frame[column].notna().any()
            and self.frame[column].nunique(dropna=True) <= 100
        )
        self.group_combo.clear()
        self.group_combo.addItem("(none)")
        self.group_combo.addItems(groups)
        self.player_controls.refresh_columns(self.frame, numeric)
        self.run_button.setEnabled(len(numeric) >= 2)
        self.source_label.setText(
            f"Source: {self.source_name} · {len(self.frame):,} rows · "
            f"{len(self.frame.columns)} columns · SHA-256: "
            f"{self.source_sha256 or 'in-memory'}"
        )
        self.last_result = None
        self.player_payload: dict[str, object] = {}
        self.result_table.clear()
        populate_table_preview(self.data_preview, self.frame)
        self.plot_widget.backing_data = pd.DataFrame()
        self._refresh_convention_evidence()
        self._refresh_guidance()

    def import_path(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        frame = load_imported_dataset(resolved)
        if len(frame) < 3 or len(numeric_columns(frame)) < 2:
            raise ValueError(
                "The file needs at least three rows and two numeric columns"
            )
        digest = file_sha256(resolved)
        self.set_frame(frame, resolved.name, "imported", digest, resolved)
        self.dataset_combo.blockSignals(True)
        imported_index = self.dataset_combo.findData("imported")
        if imported_index < 0:
            self.dataset_combo.addItem(f"Imported — {resolved.name}", "imported")
            imported_index = self.dataset_combo.count() - 1
        self.dataset_combo.setCurrentIndex(imported_index)
        self.dataset_combo.blockSignals(False)

    def import_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            cast(QWidget, self),
            "Import Launch Monitor Data",
            "",
            "Data Files (*.csv *.json)",
        )
        if selected:
            try:
                self.import_path(Path(selected))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                QMessageBox.critical(
                    cast(QWidget, self),
                    "Import Failed",
                    str(error),
                )

    def _project(self) -> AnalysisProject:
        campaign = self.catalog if self.dataset_id not in {"demo", "imported"} else None
        return AnalysisProject(
            campaign_root=str(campaign.root) if campaign else "",
            dataset_id=self.dataset_id,
            source_sha256=campaign.source_sha256 if campaign else self.source_sha256,
            selections={
                "outcome": self.outcome_combo.currentText(),
                "predictors": [
                    item.text() for item in self.predictor_list.selectedItems()
                ],
                "group": self.group_combo.currentText(),
                "plot_mode": self.player_controls.plot_mode_combo.currentText(),
                "lateral": self.player_controls.lateral_combo.currentText(),
                "carry": self.player_controls.carry_combo.currentText(),
                "session": self.player_controls.session_combo.currentText(),
                "player": self.player_controls.player_combo.currentText(),
                "time": self.player_controls.time_combo.currentText(),
                "target_distance_yd": self.player_controls.target_distance_spin.value(),
                "start_lie": self.player_controls.start_lie_combo.currentText(),
                "end_lie": self.player_controls.end_lie_combo.currentText(),
                "convention": str(self.convention_combo.currentData().value),
                "analysis_mode": self.mode_combo.currentText(),
                "correlation_method": self.method_combo.currentText(),
                "missing_policy": self.missing_combo.currentText(),
                "confidence_level": self.confidence_spin.value(),
                "minimum_sample_count": self.min_samples_spin.value(),
                **covariance_state.covariation_project_selections(self.player_controls),
            },
            dataset_sha256=self.source_sha256,
            data_path=getattr(self, "data_path", ""),
        )

    def save_project_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Save Analysis Project",
            "launch-monitor-analysis.lmproject.json",
            "Launch Monitor Project (*.lmproject.json)",
        )
        if selected:
            save_analysis_project(Path(selected), self._project())

    def load_project_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            cast(QWidget, self),
            "Load Analysis Project",
            "",
            "Launch Monitor Project (*.lmproject.json)",
        )
        if not selected:
            return
        try:
            self._load_project(load_analysis_project(Path(selected)))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.critical(
                cast(QWidget, self),
                "Project Load Failed",
                str(error),
            )

    def _load_project(self, project: AnalysisProject) -> None:
        resolved = (
            None if project.dataset_id == "demo" else resolve_project_data(project)
        )
        selections = project.selections
        confidence = selections.get("confidence_level", 0.95)
        minimum_n = selections.get("minimum_sample_count", 10)
        if not isinstance(confidence, (float, int)) or not isinstance(minimum_n, int):
            raise ValueError("saved confidence and minimum sample count are invalid")
        target = selections.get("target_distance_yd", 240.0)
        if not isinstance(target, (float, int)):
            raise ValueError("saved target distance must be numeric")
        if not 0.51 <= float(confidence) <= 0.999 or minimum_n < 3:
            raise ValueError("saved statistical thresholds are outside UI bounds")
        if not 10.0 <= float(target) <= 600.0:
            raise ValueError("saved target distance is outside UI bounds")
        if resolved and resolved.catalog and resolved.descriptor:
            self.catalog = resolved.catalog
            self.dataset_combo.blockSignals(True)
            self._populate_catalog_combo(resolved.catalog)
            self.dataset_combo.setCurrentIndex(
                self.dataset_combo.findData(project.dataset_id)
            )
            self.dataset_combo.blockSignals(False)
            self.set_frame(
                resolved.frame,
                resolved.descriptor.label,
                resolved.descriptor.dataset_id,
                resolved.descriptor.sha256,
                resolved.path,
            )
        elif project.dataset_id == "demo":
            self.load_demo()
        else:
            assert resolved is not None
            path = resolved.path
            self.set_frame(
                resolved.frame, path.name, "imported", project.dataset_sha256, path
            )
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.addItem(f"Imported — {path.name}", "imported")
            self.dataset_combo.setCurrentIndex(self.dataset_combo.count() - 1)
            self.dataset_combo.blockSignals(False)
        self.outcome_combo.setCurrentText(str(selections.get("outcome", "")))
        raw_predictors = selections.get("predictors", [])
        wanted = (
            {str(item) for item in raw_predictors}
            if isinstance(raw_predictors, list)
            else set()
        )
        for item_index in range(self.predictor_list.count()):
            item = self.predictor_list.item(item_index)
            if item is not None:
                item.setSelected(item.text() in wanted)
        combos = (
            (self.group_combo, "group"),
            (self.player_controls.plot_mode_combo, "plot_mode"),
            (self.player_controls.lateral_combo, "lateral"),
            (self.player_controls.carry_combo, "carry"),
            (self.player_controls.session_combo, "session"),
            (self.player_controls.player_combo, "player"),
            (self.player_controls.time_combo, "time"),
            (self.player_controls.start_lie_combo, "start_lie"),
            (self.player_controls.end_lie_combo, "end_lie"),
            (self.mode_combo, "analysis_mode"),
            (self.method_combo, "correlation_method"),
            (self.missing_combo, "missing_policy"),
        )
        for combo, key in combos:
            combo.setCurrentText(str(selections.get(key, combo.currentText())))
        self.player_controls.target_distance_spin.setValue(float(target))
        convention = str(selections.get("convention", ""))
        for convention_index in range(self.convention_combo.count()):
            item_value = self.convention_combo.itemData(convention_index).value
            if str(item_value) == convention:
                self.convention_combo.setCurrentIndex(convention_index)
                break
        self.confidence_spin.setValue(float(confidence))
        self.min_samples_spin.setValue(minimum_n)
        covariance_state.restore_covariation_project_selections(
            self.player_controls, selections
        )

    def export_data_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Export Retained Data",
            "launch-monitor-records.csv",
            "CSV (*.csv);;JSON (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() == ".json":
            path.write_text(
                self.frame.to_json(orient="records", indent=2),
                encoding="utf-8",
                newline="\n",
            )
        else:
            self.frame.to_csv(path, index=False, lineterminator="\n")

    def export_result_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Export Launch Monitor Analysis",
            "launch-monitor-analysis.json",
            "JSON (*.json)",
        )
        if selected:
            self._refresh_details()
            Path(selected).write_text(
                self.details.toPlainText() + "\n", encoding="utf-8", newline="\n"
            )
