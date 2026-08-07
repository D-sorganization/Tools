"""Statistical and player-analysis behavior for the PyQt workbench."""

from __future__ import annotations

import json
from typing import cast

import pandas as pd
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QWidget,
)

from rate_of_closure.launch_monitor_analysis import (
    AnalysisMode,
    AnalysisRequest,
    AnalysisResult,
    CorrelationMethod,
    MissingPolicy,
    analyze_launch_monitor_data,
)
from rate_of_closure.launch_monitor_data import CampaignDatasetCatalog
from rate_of_closure.launch_monitor_player_metrics import (
    BROADIE_SOURCE_URL,
    analyze_dispersion,
    analyze_sessions,
    calculate_strokes_gained_proxy,
)
from rate_of_closure.ui.pyqt6.launch_monitor_covariation_presenter import (
    run_covariation_presentation,
    run_covariation_scan_presentation,
)
from rate_of_closure.ui.pyqt6.launch_monitor_player_controls import (
    LaunchMonitorPlayerControls,
)
from rate_of_closure.ui.pyqt6.launch_monitor_plot_widget import (
    LaunchMonitorPlotWidget,
)
from shared.python.swing_sim.conventions import (
    ParameterId,
    convention_registry,
)


class LaunchMonitorAnalysisMixin:
    """Mixin isolated to keep the QWidget composition reviewable."""

    frame: pd.DataFrame
    source_name: str
    dataset_id: str
    source_sha256: str
    catalog: CampaignDatasetCatalog | None
    last_result: AnalysisResult | None
    player_payload: dict[str, object]
    convention_combo: QComboBox
    convention_evidence: QLabel
    outcome_combo: QComboBox
    predictor_list: QListWidget
    mode_combo: QComboBox
    method_combo: QComboBox
    missing_combo: QComboBox
    group_combo: QComboBox
    confidence_spin: QDoubleSpinBox
    min_samples_spin: QSpinBox
    result_table: QTableWidget
    plot_widget: LaunchMonitorPlotWidget
    details: QPlainTextEdit
    guidance: QTextBrowser
    player_controls: LaunchMonitorPlayerControls

    def _refresh_convention_evidence(self) -> None:
        try:
            parameter = ParameterId(self.outcome_combo.currentText())
        except ValueError:
            parameter = ParameterId.CLUB_SPEED
        definition = convention_registry().definition(
            self.convention_combo.currentData(), parameter
        )
        reference = definition.reference_point.value.replace("_", " ")
        event_time = definition.event_time.value.replace("_", " ")
        self.convention_evidence.setText(
            f"<b>{definition.label}</b>: {reference}, {event_time}. "
            f"<a href='{definition.source_url}'>Source</a>"
        )

    def _selected_predictors(self) -> tuple[str, ...]:
        return tuple(item.text() for item in self.predictor_list.selectedItems())

    def run_analysis(self) -> AnalysisResult:
        group = self.group_combo.currentText()
        result = analyze_launch_monitor_data(
            self.frame,
            AnalysisRequest(
                outcome=self.outcome_combo.currentText(),
                predictors=self._selected_predictors(),
                analysis_mode=cast(AnalysisMode, self.mode_combo.currentText()),
                correlation_method=cast(
                    CorrelationMethod, self.method_combo.currentText()
                ),
                missing_policy=cast(MissingPolicy, self.missing_combo.currentText()),
                group_by=None if group == "(none)" else group,
                confidence_level=self.confidence_spin.value(),
                min_samples=self.min_samples_spin.value(),
            ),
        )
        rows = [
            [
                item.predictor,
                "correlation",
                f"{item.coefficient:.6g}" if item.coefficient is not None else "—",
                f"{item.p_value:.6g}" if item.p_value is not None else "—",
                f"{item.adjusted_p_value:.6g}"
                if item.adjusted_p_value is not None
                else "—",
                str(item.sample_count),
            ]
            for item in result.correlations
        ]
        if result.regression:
            rows.extend(
                [
                    [
                        name,
                        "OLS coefficient",
                        f"{value.estimate:.6g}",
                        f"{value.p_value:.6g}",
                        f"[{value.ci_lower:.6g}, {value.ci_upper:.6g}]",
                        str(result.regression.sample_count),
                    ]
                    for name, value in result.regression.coefficients.items()
                ]
            )
        headers = ["Variable", "Statistic", "Estimate", "p", "Adjusted p / CI", "N"]
        self.result_table.setColumnCount(len(headers))
        self.result_table.setHorizontalHeaderLabels(headers)
        self.result_table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column_index, value in enumerate(values):
                self.result_table.setItem(
                    row_index, column_index, QTableWidgetItem(value)
                )
        self.result_table.resizeColumnsToContents()
        self.last_result = result
        self.render_selected_plot()
        self._refresh_details()
        return result

    def render_selected_plot(self) -> None:
        mode = self.player_controls.plot_mode_combo.currentText()
        predictors = self._selected_predictors()
        if mode == "Relationship":
            if not predictors:
                raise ValueError("select at least one predictor")
            self.plot_widget.plot_relationship(
                self.frame, predictors[0], self.outcome_combo.currentText()
            )
            self.player_payload = {
                "mode": mode,
                "description": self.plot_widget.description,
            }
        elif mode == "Directional Dispersion":
            analysis = analyze_dispersion(
                self.frame,
                self.player_controls.lateral_combo.currentText(),
                self.player_controls.carry_combo.currentText(),
            )
            self.plot_widget.plot_dispersion(analysis)
            self.player_payload = {
                "mode": mode,
                "summary": {
                    "n": analysis.sample_count,
                    "left": analysis.left_count,
                    "center": analysis.center_count,
                    "right": analysis.right_count,
                    "mean_lateral_yd": analysis.mean_lateral_yd,
                    "lateral_std_yd": analysis.lateral_std_yd,
                    "absolute_p50_yd": analysis.absolute_p50_yd,
                    "absolute_p80_yd": analysis.absolute_p80_yd,
                    "ellipse_major_radius_yd": analysis.ellipse_major_radius_yd,
                    "ellipse_minor_radius_yd": analysis.ellipse_minor_radius_yd,
                    "ellipse_angle_deg": analysis.ellipse_angle_deg,
                },
                "description": analysis.method_description,
            }
        elif mode == "Strokes Gained":
            self._render_strokes_gained(mode)
        elif mode == "Session Trend":
            self._render_session_trend(mode)
        elif mode == "Within-Player Covariation":
            self.player_payload = run_covariation_presentation(
                self.frame, self.player_controls, self.plot_widget, self.result_table
            )
        else:
            self.player_payload = run_covariation_scan_presentation(
                self.frame, self.player_controls, self.plot_widget, self.result_table
            )

    def _render_strokes_gained(self, mode: str) -> None:
        analysis = calculate_strokes_gained_proxy(
            self.frame,
            carry_column=self.player_controls.carry_combo.currentText(),
            lateral_column=self.player_controls.lateral_combo.currentText(),
            target_distance_yd=self.player_controls.target_distance_spin.value(),
            start_lie=self.player_controls.start_lie_combo.currentText(),
            end_lie=self.player_controls.end_lie_combo.currentText(),
        )
        self.plot_widget.plot_strokes_gained(analysis)
        self.player_payload = {
            "mode": mode,
            "summary": {
                "n": analysis.sample_count,
                "mean_strokes_gained_proxy": analysis.mean_strokes_gained_proxy,
                "median_strokes_gained_proxy": analysis.median_strokes_gained_proxy,
                "clamped_count": int(analysis.backing_data["benchmark_clamped"].sum()),
                "clamped_fraction": analysis.clamped_fraction,
            },
            "description": analysis.method_description,
            "reference_source": BROADIE_SOURCE_URL,
            "reference_table": analysis.reference_table.to_dict(orient="records"),
        }

    def _render_session_trend(self, mode: str) -> None:
        session = self.player_controls.session_combo.currentText()
        if session == "(unavailable)":
            raise ValueError("select or import a session identifier column")
        player = self.player_controls.player_combo.currentText()
        player_column = None if player == "(all players)" else player
        time = self.player_controls.time_combo.currentText()
        time_column = None if time == "(row order)" else time
        analysis = analyze_sessions(
            self.frame,
            metric_column=self.outcome_combo.currentText(),
            session_column=session,
            player_column=player_column,
            time_column=time_column,
        )
        self.plot_widget.plot_sessions(
            analysis,
            self.outcome_combo.currentText(),
            player_column=player_column,
            source_frame=self.frame,
        )
        self.player_payload = {
            "mode": mode,
            "trend_slope_per_session": analysis.trend_slope_per_session,
            "trend_slope_per_day": analysis.trend_slope_per_day,
            "metric_unit": analysis.metric_unit,
            "description": analysis.method_description,
            "session_summary": analysis.summary.to_dict(orient="records"),
        }

    def _refresh_details(self) -> None:
        payload = {
            "source": {
                "name": self.source_name,
                "dataset_id": self.dataset_id,
                "campaign_source_sha256": (
                    self.catalog.source_sha256 if self.catalog else ""
                ),
                "dataset_sha256": self.source_sha256,
                "rows": len(self.frame),
                "columns": len(self.frame.columns),
            },
            "statistics": self.last_result.to_wire() if self.last_result else None,
            "player_analysis": self.player_payload,
            "plot_backing_rows": len(self.plot_widget.backing_data),
        }
        self.details.setPlainText(json.dumps(payload, indent=2, sort_keys=True))

    def _refresh_guidance(self) -> None:
        self.guidance.setHtml(
            "<h3>Calculation guide</h3>"
            "<p><b>Relationship:</b> every finite pair; axes show source units.</p>"
            "<p><b>Dispersion:</b> negative is left and positive is right. P50/P80 "
            "are absolute lateral-error quantiles. The 80% ellipse is a covariance "
            "contour, not a guaranteed boundary.</p>"
            "<p><b>Strokes gained proxy:</b> E[before] − 1 − E[after]. Remaining "
            "distance is √((target − carry)² + lateral²). Benchmarks use "
            f"<a href='{BROADIE_SOURCE_URL}'>Broadie Table 9</a>. Assumed ending "
            "lie and missing putting transitions make this a range proxy.</p>"
            "<p><b>Session trend:</b> mean ± sample SD; trend is OLS change per "
            "displayed session, fitted independently per player. Selecting a "
            "timestamp also reports change per elapsed day. Neither adjusts for "
            "context.</p>"
            "<p><b>Within-player covariation:</b> Pearson measures linear and "
            "Spearman monotonic association. Player-mean centering separates "
            "within-player from between-player patterns. Fisher-z fixed and "
            "random effects summarize eligible player Pearson correlations; "
            "heterogeneity means players need not share one relationship. "
            "An explicitly selected identity column is required. Association "
            "does not establish causality, coaching mechanism, or vendor internals.</p>"
            "<p><b>Covariation pair scan:</b> ranks every numeric pair by absolute "
            "random-effects Pearson correlation. It is exploratory; testing many "
            "pairs creates multiplicity and false-positive risk. Validate findings "
            "on held-out data. Correlation does not imply causation.</p>"
            "<p><b>Advanced model explanations:</b> private PCA and importance "
            "tables are selectable datasets. They describe association, not "
            "causality or vendor internals.</p>"
        )

    def run_analysis_safely(self) -> None:
        try:
            self.run_analysis()
        except (KeyError, ValueError) as error:
            QMessageBox.warning(cast(QWidget, self), "Analysis Not Run", str(error))


__all__ = ["LaunchMonitorAnalysisMixin"]
