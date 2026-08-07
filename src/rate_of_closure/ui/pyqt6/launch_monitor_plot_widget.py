"""Unit-aware, exportable Matplotlib canvas for launch-monitor analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout, QWidget

from rate_of_closure.launch_monitor_data import axis_label
from rate_of_closure.launch_monitor_player_metrics import (
    DispersionAnalysis,
    SessionAnalysis,
    StrokesGainedAnalysis,
)


class LaunchMonitorPlotWidget(QWidget):
    """Render and export relationship, dispersion, SG, and trend plots."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setAccessibleName("Launch Monitor Analysis Plot")
        self.canvas.setToolTip(
            "Interactive analytical plot; axis labels always include known units"
        )
        self.backing_data = pd.DataFrame()
        self.description = "No plot has been calculated."
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def _axis(self):  # type: ignore[no-untyped-def]
        self.figure.clear()
        return self.figure.add_subplot(111)

    @staticmethod
    def _dataset_axis_label(
        frame: pd.DataFrame,
        column: str,
        retained_index: pd.Index | None = None,
    ) -> str:
        inferred = str(axis_label(column))
        if "unit" not in frame or "(" in inferred:
            return inferred
        dimensionless = {
            "n",
            "count",
            "sample_count",
            "r2",
            "normalized_rmse_sd",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p",
            "calibration_slope",
            "mean_rank",
            "win_fraction",
        }
        if column.lower() in dimensionless:
            return inferred
        unit_rows = frame if retained_index is None else frame.loc[retained_index]
        units = {
            str(value).strip()
            for value in unit_rows["unit"].dropna()
            if str(value).strip()
        }
        if len(units) > 1:
            rendered = ", ".join(sorted(units))
            raise ValueError(
                f"{column} spans mixed units ({rendered}); filter the dataset "
                "to one outcome/unit before plotting"
            )
        if not units:
            return inferred
        label = column.replace("_", " ").strip().title()
        return f"{label} ({next(iter(units))})"

    def plot_relationship(
        self, frame: pd.DataFrame, x_column: str, y_column: str
    ) -> None:
        pairs = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce")
        pairs = pairs.dropna()
        if len(pairs) < 2:
            raise ValueError("relationship plot requires at least two finite pairs")
        x_label = self._dataset_axis_label(frame, x_column, pairs.index)
        y_label = self._dataset_axis_label(frame, y_column, pairs.index)
        axis = self._axis()
        axis.scatter(
            pairs[x_column],
            pairs[y_column],
            s=8,
            alpha=0.45,
            color="#168aad",
            rasterized=True,
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(f"{y_column} versus {x_column}")
        axis.grid(alpha=0.2)
        self.backing_data = pairs.reset_index(names="source_index")
        self.description = (
            "Each point is one complete observed pair. The plot performs no "
            "aggregation or causal adjustment."
        )
        self.canvas.draw_idle()

    def plot_dispersion(self, analysis: DispersionAnalysis) -> None:
        axis = self._axis()
        backing = analysis.backing_data
        if "downrange_yd" in backing:
            axis.scatter(
                backing["lateral_yd"],
                backing["downrange_yd"],
                s=8,
                alpha=0.4,
                color="#168aad",
                rasterized=True,
            )
            axis.set_ylabel("Downrange Distance (yd)")
            if (
                analysis.ellipse_major_radius_yd is not None
                and analysis.ellipse_minor_radius_yd is not None
                and analysis.ellipse_angle_deg is not None
            ):
                ellipse = Ellipse(
                    (
                        float(backing["lateral_yd"].mean()),
                        float(backing["downrange_yd"].mean()),
                    ),
                    width=2.0 * analysis.ellipse_major_radius_yd,
                    height=2.0 * analysis.ellipse_minor_radius_yd,
                    angle=analysis.ellipse_angle_deg,
                    fill=False,
                    edgecolor="#f77f00",
                    linewidth=2.0,
                    label="80% covariance ellipse",
                )
                axis.add_patch(ellipse)
        else:
            axis.scatter(
                backing["lateral_yd"],
                backing.index,
                s=8,
                alpha=0.4,
                color="#168aad",
                rasterized=True,
            )
            axis.set_ylabel("Shot Sequence (shot)")
        axis.axvline(0.0, color="#d62828", linestyle="--", label="Target line")
        axis.set_xlabel("Lateral Outcome (yd; left − / right +)")
        axis.set_title("Directional Dispersion")
        axis.legend()
        axis.grid(alpha=0.2)
        self.backing_data = backing.assign(
            absolute_p50_yd=analysis.absolute_p50_yd,
            absolute_p80_yd=analysis.absolute_p80_yd,
            ellipse_major_radius_yd=analysis.ellipse_major_radius_yd,
            ellipse_minor_radius_yd=analysis.ellipse_minor_radius_yd,
            ellipse_angle_deg=analysis.ellipse_angle_deg,
        )
        self.description = analysis.method_description
        self.canvas.draw_idle()

    def plot_strokes_gained(self, analysis: StrokesGainedAnalysis) -> None:
        axis = self._axis()
        backing = analysis.backing_data
        scatter = axis.scatter(
            backing["source_index"],
            backing["strokes_gained_proxy"],
            c=backing["lateral_yd"],
            cmap="coolwarm",
            s=9,
            alpha=0.55,
            rasterized=True,
        )
        axis.axhline(0.0, color="black", linestyle="--")
        axis.set_xlabel("Source Shot Index (shot)")
        axis.set_ylabel("Strokes Gained Proxy (strokes/shot)")
        axis.set_title("Range-Shot Strokes Gained Proxy")
        colorbar = self.figure.colorbar(scatter, ax=axis)
        colorbar.set_label("Lateral Outcome (yd; left − / right +)")
        clamped_count = int(backing["benchmark_clamped"].sum())
        axis.text(
            0.01,
            0.99,
            (
                f"Benchmark clamped: {clamped_count}/{analysis.sample_count} "
                f"({analysis.clamped_fraction:.1%})"
            ),
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#777777"},
        )
        axis.grid(alpha=0.2)
        self.backing_data = backing.copy()
        self.description = analysis.method_description
        self.canvas.draw_idle()

    def plot_sessions(
        self,
        analysis: SessionAnalysis,
        metric_column: str,
        *,
        player_column: str | None = None,
        source_frame: pd.DataFrame | None = None,
    ) -> None:
        axis = self._axis()
        summary = analysis.summary
        player_groups = (
            summary.groupby(player_column, sort=False, dropna=False)
            if player_column and player_column in summary
            else [(None, summary)]
        )
        for player, player_sessions in player_groups:
            label = None
            if player is not None:
                slope = float(player_sessions["trend_slope_per_session"].iloc[0])
                label = f"{player} (slope {slope:.3g}/session)"
            axis.errorbar(
                player_sessions["session_sequence"],
                player_sessions["mean"],
                yerr=player_sessions["std"],
                marker="o",
                capsize=3,
                label=label,
            )
        if player_column and player_column in summary:
            axis.legend(title=player_column.replace("_", " ").title())
        axis.set_xlabel("Session Sequence (session)")
        axis.set_ylabel(
            self._dataset_axis_label(
                source_frame if source_frame is not None else summary,
                metric_column,
            )
        )
        axis.set_title("Session Mean ± One Sample Standard Deviation")
        axis.grid(alpha=0.2)
        self.backing_data = summary.copy()
        self.description = analysis.method_description
        self.canvas.draw_idle()

    def save_plot_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Launch Monitor Plot",
            "launch-monitor-plot.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if selected:
            self.figure.savefig(Path(selected), dpi=200)

    def export_backing_dialog(self) -> None:
        if self.backing_data.empty:
            QMessageBox.warning(self, "No Plot Data", "Calculate a plot first.")
            return
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot Backing Data",
            "launch-monitor-plot-data.csv",
            "CSV (*.csv);;JSON (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() == ".json":
            path.write_text(
                self.backing_data.to_json(orient="records", indent=2),
                encoding="utf-8",
                newline="\n",
            )
        else:
            self.backing_data.to_csv(path, index=False, lineterminator="\n")


__all__ = ["LaunchMonitorPlotWidget"]
