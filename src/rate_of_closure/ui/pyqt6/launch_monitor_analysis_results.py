"""Render launch-monitor statistical results without coupling analysis to widgets."""

from __future__ import annotations

import json

from PyQt6.QtWidgets import QPlainTextEdit, QTableWidget, QTableWidgetItem

from rate_of_closure.launch_monitor_analysis import AnalysisResult


def render_analysis_result(
    result: AnalysisResult, table: QTableWidget, details: QPlainTextEdit
) -> None:
    """Replace visible result evidence with one exact analysis result."""
    rows = [
        [
            item.predictor,
            "correlation",
            "—" if item.coefficient is None else f"{item.coefficient:.6g}",
            "—" if item.p_value is None else f"{item.p_value:.6g}",
            "—" if item.adjusted_p_value is None else f"{item.adjusted_p_value:.6g}",
            str(item.sample_count),
        ]
        for item in result.correlations
    ]
    if result.regression:
        rows.extend(
            [
                name,
                "OLS coefficient",
                f"{coefficient.estimate:.6g}",
                f"{coefficient.p_value:.6g}",
                f"[{coefficient.ci_lower:.6g}, {coefficient.ci_upper:.6g}]",
                str(result.regression.sample_count),
            ]
            for name, coefficient in result.regression.coefficients.items()
        )
    headers = ["Variable", "Statistic", "Estimate", "p", "Adjusted p / CI", "N"]
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(rows))
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            table.setItem(row_index, column_index, QTableWidgetItem(value))
    table.resizeColumnsToContents()
    details.setPlainText(json.dumps(result.to_wire(), indent=2, sort_keys=True))


__all__ = ["render_analysis_result"]
