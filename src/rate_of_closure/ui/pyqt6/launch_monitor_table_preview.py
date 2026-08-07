"""Bounded table preview that retains the complete source frame elsewhere."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem


def populate_table_preview(
    table: QTableWidget, frame: pd.DataFrame, max_rows: int = 500
) -> None:
    """Show a bounded preview without mutating or truncating the source frame."""

    columns = [str(column) for column in frame.columns]
    preview = frame.head(max_rows)
    table.clear()
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setRowCount(len(preview))
    for row_index, values in enumerate(preview.itertuples(index=False, name=None)):
        for column_index, value in enumerate(values):
            rendered = "" if pd.isna(value) else str(value)
            table.setItem(row_index, column_index, QTableWidgetItem(rendered))
    table.resizeColumnsToContents()
    table.setToolTip(
        f"Showing {len(preview):,} of {len(frame):,} retained rows; exports and "
        "calculations use the complete dataset"
    )


__all__ = ["populate_table_preview"]
