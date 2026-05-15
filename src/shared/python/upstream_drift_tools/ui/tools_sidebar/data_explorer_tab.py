"""Qt Data Explorer tab for the shared Sidekick sidebar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_explorer_service import (
    DEFAULT_DATA_EXPLORER_MAX_FILE_SIZE_BYTES,
    DEFAULT_DATA_EXPLORER_PREVIEW_ROWS,
    DataExplorerError,
    DataExplorerPreview,
    DataExplorerService,
)
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .qt_compat import QtWidgets
from .settings import SidebarTabSettingsDescriptor, SidebarTabSettingsSchema

DATA_EXPLORER_TAB_ID = "data_explorer"
DATA_EXPLORER_TAB_SETTINGS = SidebarTabSettingsDescriptor(
    schema=SidebarTabSettingsSchema(
        version=1,
        defaults={
            "preview_rows": DEFAULT_DATA_EXPLORER_PREVIEW_ROWS,
            "max_file_size_mb": DEFAULT_DATA_EXPLORER_MAX_FILE_SIZE_BYTES
            // (1024 * 1024),
        },
        allowed_keys=frozenset({"preview_rows", "max_file_size_mb"}),
    )
)


def build_data_explorer_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the lightweight Data Explorer tab."""
    widget = SidekickDataExplorerWidget(sidebar)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP[DATA_EXPLORER_TAB_ID]["summary"])
    return widget


class SidekickDataExplorerWidget(QtWidgets.QWidget):
    """Small bounded data-file preview surface for Sidekick."""

    def __init__(self, sidebar: Any) -> None:
        super().__init__(sidebar)
        self.setObjectName("SidekickDataExplorerTab")
        self._sidebar = sidebar
        self._preview: DataExplorerPreview | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        path_row = QtWidgets.QHBoxLayout()
        self._path_input = QtWidgets.QLineEdit(self)
        self._path_input.setObjectName("SidekickDataExplorerPath")
        self._path_input.setPlaceholderText(
            "project-relative or absolute data file path"
        )
        self._path_input.setToolTip("Enter a CSV, TSV, JSON, Parquet, or Excel file.")
        path_row.addWidget(self._path_input, stretch=1)

        browse = QtWidgets.QPushButton("Browse", self)
        browse.setObjectName("SidekickDataExplorerBrowse")
        browse.setToolTip("Choose a project-scoped data file.")
        browse.clicked.connect(self._choose_file)
        path_row.addWidget(browse)

        self._load_button = QtWidgets.QPushButton("Load Preview", self)
        self._load_button.setObjectName("SidekickDataExplorerLoad")
        self._load_button.setToolTip("Preview the file using bounded Sidekick limits.")
        self._load_button.clicked.connect(self.load_preview)
        path_row.addWidget(self._load_button)
        layout.addLayout(path_row)

        self._status = QtWidgets.QLabel(
            DEFAULT_SIDEBAR_TAB_HELP[DATA_EXPLORER_TAB_ID]["summary"],
            self,
        )
        self._status.setObjectName("SidekickDataExplorerStatus")
        self._status.setWordWrap(True)
        self._status.setToolTip(
            "Shows preview mode, schema size, and validation errors."
        )
        layout.addWidget(self._status)

        self._preview_table = QtWidgets.QTableWidget(self)
        self._preview_table.setObjectName("SidekickDataExplorerPreviewTable")
        self._preview_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._preview_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self._preview_table, stretch=1)

        options_row = QtWidgets.QHBoxLayout()
        self._columns_input = QtWidgets.QLineEdit(self)
        self._columns_input.setObjectName("SidekickDataExplorerColumns")
        self._columns_input.setPlaceholderText("column_a, column_b (blank = all)")
        self._columns_input.setToolTip(
            "Comma-separated columns to export or hand off to Data Processor."
        )
        options_row.addWidget(self._columns_input, stretch=1)

        self._variable_input = QtWidgets.QLineEdit(self)
        self._variable_input.setObjectName("SidekickDataExplorerVariable")
        self._variable_input.setPlaceholderText("workspace_variable_name")
        self._variable_input.setToolTip(
            "Workspace variable name for exported preview data."
        )
        options_row.addWidget(self._variable_input, stretch=1)
        layout.addLayout(options_row)

        action_row = QtWidgets.QHBoxLayout()
        self._export_button = QtWidgets.QPushButton("Export Preview", self)
        self._export_button.setObjectName("SidekickDataExplorerExport")
        self._export_button.setToolTip(
            "Export the selected preview columns into the shared Sidekick workspace."
        )
        self._export_button.clicked.connect(self.export_preview)
        action_row.addWidget(self._export_button)

        self._handoff_button = QtWidgets.QPushButton("Send to Data Processor", self)
        self._handoff_button.setObjectName("SidekickDataExplorerSendToDataProcessor")
        self._handoff_button.setToolTip(
            "Emit a structured Data Processor launch request for the active preview."
        )
        self._handoff_button.clicked.connect(self.send_to_data_processor)
        action_row.addWidget(self._handoff_button)
        layout.addLayout(action_row)

    def load_preview(self) -> None:
        """Load the current file path into the preview table."""
        try:
            preview = self._service().preview_file(self._path_input.text().strip())
        except DataExplorerError as exc:
            self._preview = None
            self._status.setText(str(exc))
            self._preview_table.setRowCount(0)
            self._preview_table.setColumnCount(0)
            return

        self._preview = preview
        self._variable_input.setText(Path(preview.source_path).stem + "_preview")
        self._populate_preview_table(preview)
        row_count = preview.total_rows if preview.total_rows is not None else "unknown"
        self._status.setText(
            f"{preview.format.upper()} preview loaded: {row_count} rows, "
            f"{preview.total_columns} columns ({preview.load_mode})."
        )

    def export_preview(self) -> None:
        """Export the current preview selection to the shared workspace."""
        if self._preview is None:
            self._status.setText("Load a preview before exporting.")
            return
        try:
            variable = self._service().export_selection(
                self._preview,
                self._sidebar.registry,
                self._variable_input.text(),
                selected_columns=self._selected_columns(),
            )
        except DataExplorerError as exc:
            self._status.setText(str(exc))
            return

        self._sidebar.set_context_variable(
            variable.name,
            self._sidebar.registry.get(variable.name),
        )
        self._status.setText(
            f"Exported preview to workspace variable '{variable.name}'."
        )

    def send_to_data_processor(self) -> None:
        """Emit a structured request for a future Data Processor host handoff."""
        if self._preview is None:
            self._status.setText("Load a preview before sending to Data Processor.")
            return
        try:
            payload = self._service().build_data_processor_request(
                self._preview,
                selected_columns=self._selected_columns(),
            )
        except DataExplorerError as exc:
            self._status.setText(str(exc))
            return
        self._sidebar.request_tool_launch("data_processor", payload)
        self._status.setText("Prepared Data Processor handoff request.")

    def _choose_file(self) -> None:
        filename, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose a data file",
            str(self._sidebar.project_root),
            "Data Files (*.csv *.tsv *.json *.parquet *.xls *.xlsx)",
        )
        if filename:
            self._path_input.setText(filename)

    def _populate_preview_table(self, preview: DataExplorerPreview) -> None:
        self._preview_table.setColumnCount(len(preview.columns))
        self._preview_table.setHorizontalHeaderLabels(
            [column.name for column in preview.columns]
        )
        self._preview_table.setRowCount(len(preview.preview_rows))
        for row_index, row in enumerate(preview.preview_rows):
            for column_index, column in enumerate(preview.columns):
                value = row.get(column.name)
                text = "" if value is None else str(value)
                self._preview_table.setItem(
                    row_index,
                    column_index,
                    QtWidgets.QTableWidgetItem(text),
                )

    def _selected_columns(self) -> list[str] | None:
        raw = self._columns_input.text().strip()
        if not raw:
            return None
        return [column.strip() for column in raw.split(",") if column.strip()]

    def _service(self) -> DataExplorerService:
        settings = self._sidebar.tab_settings(DATA_EXPLORER_TAB_ID)["values"]
        preview_rows = int(settings["preview_rows"])
        max_file_size_bytes = int(settings["max_file_size_mb"]) * 1024 * 1024
        return DataExplorerService(
            project_root=self._sidebar.project_root,
            preview_rows=preview_rows,
            max_file_size_bytes=max_file_size_bytes,
        )
