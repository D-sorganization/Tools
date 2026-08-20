"""Unit-aware PyQt presentation for player covariation analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QFileDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure._player_covariation_types import PlayerCovariationAnalysis
from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas


def _text(value: object) -> str:
    if value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(float(value))
    ):
        return "—"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


class LaunchMonitorCovariationView(QWidget):
    """Render centered shots and player effects from the shared core."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 4), constrained_layout=True)
        self.axes = tuple(self.figure.subplots(1, 2))
        self.canvas = LifecycleSafeFigureCanvas(self.figure)
        self.canvas.setAccessibleName("Player covariation plots")
        self.table = QTableWidget()
        self.table.setAccessibleName("Per-player covariation estimates")
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.save_button = QPushButton("Save Covariation Plot...")
        self.save_button.setAccessibleName("Save player covariation plot")
        self.save_button.setToolTip("Save the centered scatter and player forest plot")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_dialog)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.save_button)
        layout.addWidget(self.table)

    def populate(self, analysis: PlayerCovariationAnalysis) -> None:
        """Replace both plots and the inspectable per-player table."""

        self._render_scatter(self.axes[0], analysis)
        self._render_forest(self.axes[1], analysis.per_player)
        self._render_table(analysis.per_player)
        self.canvas.draw()
        self.save_button.setEnabled(True)

    @staticmethod
    def _render_scatter(axis: Axes, analysis: PlayerCovariationAnalysis) -> None:
        axis.clear()
        backing = analysis.backing_data
        for player, rows in backing.groupby("player_id", sort=True):
            axis.scatter(
                rows["centered_x"],
                rows["centered_y"],
                s=18,
                alpha=0.7,
                label=str(player),
            )
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.axvline(0, color="0.5", linewidth=0.8)
        axis.set_xlabel(
            f"Player-centered {analysis.request.x_column} ({analysis.units['x']})"
        )
        axis.set_ylabel(
            f"Player-centered {analysis.request.y_column} ({analysis.units['y']})"
        )
        axis.set_title("Within-player centered shots")
        if backing["player_id"].nunique() <= 12:
            axis.legend(fontsize=7)

    @staticmethod
    def _render_forest(axis: Axes, per_player: pd.DataFrame) -> None:
        axis.clear()
        valid = per_player[per_player["pearson_r"].notna()].reset_index(drop=True)
        positions = np.arange(len(valid))
        axis.errorbar(
            valid["pearson_r"],
            positions,
            xerr=np.vstack(
                (
                    valid["pearson_r"] - valid["ci_lower"],
                    valid["ci_upper"] - valid["pearson_r"],
                )
            ),
            fmt="o",
            capsize=3,
        ) if len(valid) else None
        axis.axvline(0, color="0.5", linewidth=0.8)
        axis.set_yticks(positions, valid["player_id"].astype(str))
        axis.set_xlabel("Pearson r (unitless)")
        axis.set_title("Per-player 95% Fisher intervals")
        axis.set_xlim(-1.05, 1.05)

    def _render_table(self, rows: pd.DataFrame) -> None:
        columns = (
            "player_id",
            "sample_count",
            "pearson_r",
            "spearman_r",
            "ci_lower",
            "ci_upper",
            "slope",
            "status",
        )
        headers = (
            "Player",
            "N",
            "Pearson r",
            "Spearman ρ",
            "CI lower",
            "CI upper",
            "Slope",
            "Status",
        )
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        for row_index, (_, row) in enumerate(rows.iterrows()):
            for column_index, column in enumerate(columns):
                self.table.setItem(
                    row_index, column_index, QTableWidgetItem(_text(row[column]))
                )
        self.table.resizeColumnsToContents()

    def populate_scan(self, ranking: pd.DataFrame) -> None:
        """Render the exploratory all-pairs ranking without causal styling."""

        for axis in self.axes:
            axis.clear()
        top = ranking.head(15).iloc[::-1]
        labels = top["x_column"].astype(str) + " × " + top["y_column"].astype(str)
        self.axes[0].barh(labels, top["absolute_random_effect_r"])
        self.axes[0].set_xlabel("Absolute random-effects Pearson r (unitless)")
        self.axes[0].set_title("Exploratory pair ranking")
        self.axes[1].axis("off")
        columns = (
            "x_column",
            "y_column",
            "random_effect_r",
            "within_player_r",
            "between_player_r",
            "contributor_count",
            "i_squared_pct",
        )
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(
            ("X", "Y", "Random r", "Within r", "Between r", "Players", "I² (%)")
        )
        self.table.setRowCount(len(ranking))
        for row_index, (_, row) in enumerate(ranking.iterrows()):
            for column_index, column in enumerate(columns):
                self.table.setItem(
                    row_index, column_index, QTableWidgetItem(_text(row[column]))
                )
        self.table.resizeColumnsToContents()
        self.canvas.draw()
        self.save_button.setEnabled(True)

    def save_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Player Covariation Plot",
            "player-covariation.svg",
            "SVG (*.svg);;PNG (*.png);;PDF (*.pdf)",
        )
        if selected:
            self.figure.savefig(Path(selected), dpi=180)


__all__ = ["LaunchMonitorCovariationView"]
