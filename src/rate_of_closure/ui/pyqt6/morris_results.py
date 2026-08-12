"""Accessible table rendering for selected Morris report targets."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem

from rate_of_closure.application.morris.presentation import MorrisReportPresentation

RESULT_HEADERS = (
    "Rank",
    "Factor",
    "μ*",
    "SE(μ*)",
    "μ",
    "σ",
    "Scientific state",
    "Pair coverage",
)


def render_morris_report(
    presentation: MorrisReportPresentation,
    table: QTableWidget,
    target_detail: QLabel,
) -> None:
    """Render one target-scoped ranking with explicit data-quality denominators."""
    if not isinstance(presentation, MorrisReportPresentation):
        raise TypeError("presentation must be a MorrisReportPresentation")
    target = presentation.target
    details = [target.kind, target.unit]
    if target.coordinate_frame:
        details.append(target.coordinate_frame)
    if target.point_id:
        details.append(target.point_id)
    if target.time_s is not None:
        details.append(f"t={target.time_s:g} s")
    target_detail.setText(" · ".join(details))
    table.setRowCount(len(presentation.rows))
    for index, row in enumerate(presentation.rows):
        coverage = (
            f"{row.valid_pairs}/{row.total_pairs} valid · typed misses "
            f"{row.typed_no_impact_pairs} · unavailable misses "
            f"{row.no_impact_unavailable_pairs} · failed {row.failed_pairs} · "
            f"nonfinite {row.nonfinite_pairs}"
        )
        values = (
            "—" if row.rank is None else str(row.rank),
            f"{row.label} [{row.source_unit}]",
            _metric(row.mu_star),
            _metric(row.mu_star_standard_error),
            _metric(row.mu),
            _metric(row.sigma),
            f"{row.availability} · {row.sample_adequacy}",
            coverage,
        )
        for column, value in enumerate(values):
            table.setItem(index, column, QTableWidgetItem(value))
    table.resizeColumnsToContents()


def _metric(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.6g}"


__all__ = ["RESULT_HEADERS", "render_morris_report"]
