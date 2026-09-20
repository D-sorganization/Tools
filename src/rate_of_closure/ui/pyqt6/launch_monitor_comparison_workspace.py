"""Side-by-side launch-monitor convention comparison workspace (PyQt6)."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.python.swing_sim.conventions import (
    ConventionId,
    ParameterGroup,
    ParameterId,
    compare_definitions,
    convention_registry,
    parameter_group,
    parameter_group_label,
)


@dataclass(frozen=True)
class ComparisonRow:
    """Evaluated side-by-side comparison for one parameter."""

    parameter_id: str
    label: str
    group: str
    group_label: str
    unit: str
    trackman_value: float | None
    foresight_value: float | None
    difference: float | None
    difference_text: str
    is_comparable: bool
    reasons: tuple[str, ...]
    reasons_text: str
    trackman_ref: str
    foresight_ref: str
    trackman_time: str
    foresight_time: str
    trackman_status: str
    foresight_status: str
    definition: str


def build_comparison_rows(
    trackman_values: dict[str, float] | None = None,
    foresight_values: dict[str, float] | None = None,
) -> list[ComparisonRow]:
    """Build immutable side-by-side comparison rows across all parameters."""
    tm_vals = trackman_values or {}
    fs_vals = foresight_values or {}
    registry = convention_registry()

    from shared.python.swing_sim.conventions._catalog_data import IDENTITIES

    rows: list[ComparisonRow] = []
    for param_id in ParameterId:
        tm_def = registry.definition(ConventionId.TRACKMAN_COMPARABLE, param_id)
        fs_def = registry.definition(ConventionId.FORESIGHT_COMPARABLE, param_id)
        compat = compare_definitions(tm_def, fs_def)
        grp = parameter_group(param_id)
        grp_lbl = parameter_group_label(grp)
        identity = IDENTITIES[param_id]

        tm_val = tm_vals.get(param_id.value)
        fs_val = fs_vals.get(param_id.value)

        if compat.comparable:
            if (
                tm_val is not None
                and fs_val is not None
                and pd.notna(tm_val)
                and pd.notna(fs_val)
            ):
                diff = float(tm_val) - float(fs_val)
                diff_text = f"{diff:+.2f}"
            else:
                diff = None
                diff_text = "—"
            reasons_tuple: tuple[str, ...] = ()
            reasons_str = "—"
        else:
            diff = None
            reasons_tuple = tuple(r.value for r in compat.reasons)
            reasons_str = ", ".join(reasons_tuple)
            diff_text = f"Not comparable ({reasons_str})"

        rows.append(
            ComparisonRow(
                parameter_id=param_id.value,
                label=tm_def.label,
                group=grp.value,
                group_label=grp_lbl,
                unit=tm_def.unit,
                trackman_value=(
                    float(tm_val) if tm_val is not None and pd.notna(tm_val) else None
                ),
                foresight_value=(
                    float(fs_val) if fs_val is not None and pd.notna(fs_val) else None
                ),
                difference=diff,
                difference_text=diff_text,
                is_comparable=compat.comparable,
                reasons=reasons_tuple,
                reasons_text=reasons_str,
                trackman_ref=tm_def.reference_point.value.replace("_", " "),
                foresight_ref=fs_def.reference_point.value.replace("_", " "),
                trackman_time=tm_def.event_time.value.replace("_", " "),
                foresight_time=fs_def.event_time.value.replace("_", " "),
                trackman_status=tm_def.quantity_status.value.replace("_", " "),
                foresight_status=fs_def.quantity_status.value.replace("_", " "),
                definition=identity.definition,
            )
        )
    return rows


class LaunchMonitorComparisonWorkspace(QWidget):
    """Side-by-side TrackMan vs Foresight comparison workspace."""

    COLUMNS = (
        "Group",
        "Parameter",
        "TrackMan",
        "Foresight",
        "Signed Difference (TM - FS)",
        "Unit",
        "TM Ref Point",
        "FS Ref Point",
        "TM Event Time",
        "FS Event Time",
        "Comparability / Reason",
        "Definition",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._all_rows: list[ComparisonRow] = build_comparison_rows()
        self._filtered_rows: list[ComparisonRow] = list(self._all_rows)
        self._source_name: str = "Demo Comparison"
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        heading_box = QGroupBox(
            "Launch-Monitor Convention Comparison (TrackMan vs Foresight)"
        )
        heading_layout = QVBoxLayout(heading_box)
        note = QLabel(
            "Direct delta calculation is allowed only when parameter contracts match. "
            "Non-equivalent quantities display the typed comparability mismatch reason."
        )
        note.setWordWrap(True)
        heading_layout.addWidget(note)

        # Filter and Search toolbar
        toolbar = QHBoxLayout()
        self.group_combo = QComboBox()
        self.group_combo.setAccessibleName("Filter Parameter Group")
        self.group_combo.setToolTip("Filter parameter rows by logical category")
        self.group_combo.addItem("All Groups", "all")
        for grp in ParameterGroup:
            self.group_combo.addItem(parameter_group_label(grp), grp.value)
        self.group_combo.currentIndexChanged.connect(self._apply_filter)
        toolbar.addWidget(QLabel("Group:"))
        toolbar.addWidget(self.group_combo)

        self.search_edit = QLineEdit()
        self.search_edit.setAccessibleName("Search Comparison Parameters")
        self.search_edit.setToolTip("Search parameters by name, ID, or definition text")
        self.search_edit.setPlaceholderText("Search parameters...")
        self.search_edit.textChanged.connect(self._apply_filter)
        toolbar.addWidget(QLabel("Search:"))
        toolbar.addWidget(self.search_edit, 1)

        self.export_json_btn = QPushButton("Export JSON...")
        self.export_json_btn.setAccessibleName("Export Comparison as JSON")
        self.export_json_btn.setToolTip(
            "Export side-by-side comparison data and verdicts as JSON"
        )
        self.export_json_btn.clicked.connect(self.export_json_dialog)
        toolbar.addWidget(self.export_json_btn)

        self.export_csv_btn = QPushButton("Export CSV...")
        self.export_csv_btn.setAccessibleName("Export Comparison as CSV")
        self.export_csv_btn.setToolTip("Export comparison table rows as CSV")
        self.export_csv_btn.clicked.connect(self.export_csv_dialog)
        toolbar.addWidget(self.export_csv_btn)

        heading_layout.addLayout(toolbar)
        layout.addWidget(heading_box)

        # Table
        self.table = QTableWidget(len(self._filtered_rows), len(self.COLUMNS))
        self.table.setAccessibleName("Launch Monitor Side-by-Side Comparison Table")
        self.table.setToolTip(
            "Side-by-side comparison between TrackMan-Comparable "
            "and Foresight-Comparable conventions"
        )
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(11, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self._populate_table()

    def set_dataset(
        self,
        frame: pd.DataFrame,
        source_name: str = "In-Memory Data",
        numeric_fields: list[str] | None = None,
    ) -> None:
        """Extract TrackMan and Foresight values from the dataset if present."""
        self._source_name = source_name
        tm_vals: dict[str, float] = {}
        fs_vals: dict[str, float] = {}

        if "monitor_vendor" in frame.columns:
            tm_mask = (
                frame["monitor_vendor"].astype(str).str.lower().str.contains("trackman")
            )
            fs_mask = (
                frame["monitor_vendor"]
                .astype(str)
                .str.lower()
                .str.contains("foresight")
            )
            tm_sub = frame[tm_mask]
            fs_sub = frame[fs_mask]
            for col in frame.columns:
                if pd.api.types.is_numeric_dtype(frame[col]):
                    if not tm_sub.empty and pd.notna(tm_sub[col].mean()):
                        tm_vals[col] = float(tm_sub[col].mean())
                    if not fs_sub.empty and pd.notna(fs_sub[col].mean()):
                        fs_vals[col] = float(fs_sub[col].mean())
        else:
            for col in frame.columns:
                if pd.api.types.is_numeric_dtype(frame[col]) and pd.notna(
                    frame[col].mean()
                ):
                    mean_val = float(frame[col].mean())
                    tm_vals[col] = mean_val
                    fs_vals[col] = mean_val

        self._all_rows = build_comparison_rows(tm_vals, fs_vals)
        self._apply_filter()

    def _apply_filter(self) -> None:
        selected_grp = self.group_combo.currentData()
        query = self.search_edit.text().strip().lower()

        filtered: list[ComparisonRow] = []
        for row in self._all_rows:
            if selected_grp != "all" and row.group != selected_grp:
                continue
            if query and not (
                query in row.label.lower()
                or query in row.parameter_id.lower()
                or query in row.definition.lower()
                or query in row.group_label.lower()
            ):
                continue
            filtered.append(row)

        self._filtered_rows = filtered
        self._populate_table()

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self._filtered_rows))
        align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        align_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        for row_idx, row in enumerate(self._filtered_rows):
            tm_str = (
                f"{row.trackman_value:.2f}" if row.trackman_value is not None else "—"
            )
            fs_str = (
                f"{row.foresight_value:.2f}" if row.foresight_value is not None else "—"
            )

            items = [
                (row.group_label, align_left),
                (row.label, align_left),
                (tm_str, align_right),
                (fs_str, align_right),
                (
                    row.difference_text,
                    align_left if not row.is_comparable else align_right,
                ),
                (row.unit, align_left),
                (row.trackman_ref, align_left),
                (row.foresight_ref, align_left),
                (row.trackman_time, align_left),
                (row.foresight_time, align_left),
                (
                    (
                        "Comparable"
                        if row.is_comparable
                        else f"Incompatible: {row.reasons_text}"
                    ),
                    align_left,
                ),
                (row.definition, align_left),
            ]

            for col_idx, (text, alignment) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(alignment)
                item.setToolTip(
                    f"{self.COLUMNS[col_idx]}: {text}\nDefinition: {row.definition}"
                )
                if col_idx == 4 and not row.is_comparable:
                    font = item.font()
                    font.setItalic(True)
                    item.setFont(font)
                self.table.setItem(row_idx, col_idx, item)

    def export_json(self) -> dict[str, Any]:
        """Export current comparison state as a deterministic JSON structure."""
        return {
            "schema_version": "launch-monitor-comparison/v1",
            "source_name": self._source_name,
            "rows": [asdict(r) for r in self._filtered_rows],
            "total_count": len(self._filtered_rows),
        }

    def export_csv(self) -> str:
        """Export current filtered comparison table as CSV text."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(self.COLUMNS)
        for row in self._filtered_rows:
            tm_str = (
                f"{row.trackman_value:.4f}" if row.trackman_value is not None else ""
            )
            fs_str = (
                f"{row.foresight_value:.4f}" if row.foresight_value is not None else ""
            )
            diff_str = (
                f"{row.difference:.4f}"
                if row.difference is not None
                else row.difference_text
            )
            writer.writerow(
                [
                    row.group_label,
                    row.label,
                    tm_str,
                    fs_str,
                    diff_str,
                    row.unit,
                    row.trackman_ref,
                    row.foresight_ref,
                    row.trackman_time,
                    row.foresight_time,
                    (
                        "Comparable"
                        if row.is_comparable
                        else f"Incompatible: {row.reasons_text}"
                    ),
                    row.definition,
                ]
            )
        return buf.getvalue()

    def export_json_dialog(self) -> None:
        """Present file save dialog for JSON comparison export."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Comparison JSON",
            "launch_monitor_comparison.json",
            "JSON (*.json)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(self.export_json(), file, indent=2)

    def export_csv_dialog(self) -> None:
        """Present file save dialog for CSV comparison export."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Comparison CSV",
            "launch_monitor_comparison.csv",
            "CSV (*.csv)",
        )
        if path:
            with open(path, "w", encoding="utf-8", newline="") as file:
                file.write(self.export_csv())
