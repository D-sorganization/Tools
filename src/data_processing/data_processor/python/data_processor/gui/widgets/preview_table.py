"""Data preview table widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import pandas as pd


class PreviewTable(QWidget):
    """Widget for displaying data preview in a table."""

    MAX_PREVIEW_ROWS = 100

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the preview table."""
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        self._add_title(layout)
        self._add_table(layout)
        self._add_info_bar(layout)

    def _add_title(self, layout: QVBoxLayout) -> None:
        """Add title label."""
        title = QLabel("Data Preview")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

    def _add_table(self, layout: QVBoxLayout) -> None:
        """Add table widget."""
        self.table_widget = QTableWidget()
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table_widget)

    def _add_info_bar(self, layout: QVBoxLayout) -> None:
        """Add information bar."""
        info_layout = QHBoxLayout()

        self.row_count_label = QLabel("Rows: 0")
        self.column_count_label = QLabel("Columns: 0")

        info_layout.addWidget(self.row_count_label)
        info_layout.addWidget(self.column_count_label)
        info_layout.addStretch()

        layout.addLayout(info_layout)

    def set_data(self, df: pd.DataFrame) -> None:
        """Set data to display in the table."""
        self._clear_table()
        self._set_headers(df.columns.tolist())
        self._populate_rows(df)
        self._update_info_labels(df)

    def _clear_table(self) -> None:
        """Clear the table."""
        self.table_widget.clear()
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)

    def _set_headers(self, columns: list[str]) -> None:
        """Set table column headers."""
        self.table_widget.setColumnCount(len(columns))
        self.table_widget.setHorizontalHeaderLabels(columns)

    def _populate_rows(self, df: pd.DataFrame) -> None:
        """Populate table rows from dataframe."""
        row_count = min(len(df), self.MAX_PREVIEW_ROWS)
        self.table_widget.setRowCount(row_count)

        for row_idx in range(row_count):
            self._populate_row(row_idx, df.iloc[row_idx])

    def _populate_row(self, row_idx: int, row_data: Any) -> None:
        """Populate a single row."""
        for col_idx, value in enumerate(row_data):
            item = QTableWidgetItem(self._format_value(value))
            self.table_widget.setItem(row_idx, col_idx, item)

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _update_info_labels(self, df: pd.DataFrame) -> None:
        """Update information labels."""
        total_rows = len(df)
        displayed = min(total_rows, self.MAX_PREVIEW_ROWS)
        self.row_count_label.setText(f"Rows: {displayed}/{total_rows}")
        self.column_count_label.setText(f"Columns: {len(df.columns)}")

    def clear(self) -> None:
        """Clear the table."""
        self._clear_table()
        self.row_count_label.setText("Rows: 0")
        self.column_count_label.setText("Columns: 0")

    def get_column_count(self) -> int:
        """Get number of columns."""
        return self.table_widget.columnCount()

    def get_row_count(self) -> int:
        """Get number of rows displayed."""
        return self.table_widget.rowCount()
