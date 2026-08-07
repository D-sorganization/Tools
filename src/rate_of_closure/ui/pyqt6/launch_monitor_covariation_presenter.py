"""PyQt presentation adapter for shared player-covariation calculations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

from rate_of_closure.launch_monitor_analysis import numeric_columns
from rate_of_closure.player_covariation import (
    CovariationRequest,
    PairScanRequest,
    analyze_player_covariation,
    scan_covariation_pairs,
)
from rate_of_closure.ui.pyqt6.launch_monitor_covariation_scan_plot import (
    plot_covariation_scan,
)
from rate_of_closure.ui.pyqt6.launch_monitor_player_controls import (
    LaunchMonitorPlayerControls,
)
from rate_of_closure.ui.pyqt6.launch_monitor_plot_widget import (
    LaunchMonitorPlotWidget,
)


def run_covariation_presentation(
    frame: pd.DataFrame,
    controls: LaunchMonitorPlayerControls,
    plot_widget: LaunchMonitorPlotWidget,
    result_table: QTableWidget,
) -> dict[str, object]:
    """Calculate, render, and serialize one selected player-level analysis."""

    player = controls.player_combo.currentText()
    if player == "(all players)":
        raise ValueError("select an explicit player identifier column")
    request = CovariationRequest(
        x_column=controls.covariation_x_combo.currentText(),
        y_column=controls.covariation_y_combo.currentText(),
        player_column=player,
        min_samples=controls.covariation_min_samples_spin.value(),
        confidence_level=controls.covariation_confidence_spin.value(),
    )
    analysis = analyze_player_covariation(frame, request)
    method = controls.covariation_method_combo.currentText()
    plot_widget.plot_player_covariation(analysis, method)
    _populate_results(result_table, analysis.per_player, method)
    return _analysis_payload(analysis, method)


def _populate_results(
    table: QTableWidget, per_player: pd.DataFrame, method: str
) -> None:
    coefficient = "pearson_r" if method == "Pearson" else "spearman_r"
    headers = ["Player", "N", f"{method} r", "Pearson CI", "Slope", "R²", "Status"]
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(per_player))
    for row_index, (_, row) in enumerate(per_player.iterrows()):
        interval = _format_interval(row["ci_lower"], row["ci_upper"])
        values = (
            row["player_id"],
            row["sample_count"],
            row[coefficient],
            interval,
            row["slope"],
            row["r_squared"],
            row["status"],
        )
        for column_index, value in enumerate(values):
            table.setItem(row_index, column_index, QTableWidgetItem(_format(value)))
    table.resizeColumnsToContents()


def _format(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.5g}" if isinstance(value, float) else str(value)


def _format_interval(lower: Any, upper: Any) -> str:
    if pd.isna(lower) or pd.isna(upper):
        return "—"
    return f"[{float(lower):.5g}, {float(upper):.5g}]"


def _analysis_payload(analysis: Any, method: str) -> dict[str, object]:
    warnings = list(analysis.warnings)
    if method == "Spearman":
        warnings.append("Fisher intervals and meta-analysis summarize Pearson r only.")
    return {
        "mode": "Within-Player Covariation",
        "selected_display_method": method,
        "request": asdict(analysis.request),
        "units": analysis.units,
        "definitions": analysis.definitions,
        "pooled": asdict(analysis.pooled),
        "within_player": asdict(analysis.within_player),
        "between_player": asdict(analysis.between_player),
        "meta_analysis": asdict(analysis.meta_analysis),
        "per_player": analysis.per_player.to_dict(orient="records"),
        "backing_data": analysis.backing_data.to_dict(orient="records"),
        "warnings": warnings,
        "description": analysis.method_description,
    }


def run_covariation_scan_presentation(
    frame: pd.DataFrame,
    controls: LaunchMonitorPlayerControls,
    plot_widget: LaunchMonitorPlotWidget,
    result_table: QTableWidget,
) -> dict[str, object]:
    """Rank, render, and serialize every numeric pair for explicit identities."""

    player = controls.player_combo.currentText()
    if player == "(all players)":
        raise ValueError("select an explicit player identifier column")
    request = PairScanRequest(
        player_column=player,
        numeric_columns=tuple(
            column for column in numeric_columns(frame) if column != player
        ),
        min_samples=controls.covariation_min_samples_spin.value(),
        confidence_level=controls.covariation_confidence_spin.value(),
    )
    analysis = scan_covariation_pairs(frame, request)
    plot_covariation_scan(plot_widget, analysis.ranking)
    _populate_scan_results(result_table, analysis.ranking)
    return {
        "mode": "Covariation Pair Scan",
        "request": asdict(request),
        "pair_count": len(analysis.ranking),
        "ranking": analysis.ranking.to_dict(orient="records"),
        "warnings": list(analysis.warnings),
        "description": analysis.method_description,
    }


def _populate_scan_results(table: QTableWidget, ranking: pd.DataFrame) -> None:
    columns = (
        "x_column",
        "y_column",
        "random_effect_r",
        "within_player_r",
        "between_player_r",
        "contributor_count",
        "i_squared_pct",
    )
    headers = ("X", "Y", "Random r", "Within r", "Between r", "Players", "I² (%)")
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(ranking))
    for row_index, (_, row) in enumerate(ranking.iterrows()):
        for column_index, column in enumerate(columns):
            item = QTableWidgetItem(_format(row[column]))
            table.setItem(row_index, column_index, item)
    table.resizeColumnsToContents()


__all__ = ["run_covariation_presentation", "run_covariation_scan_presentation"]
