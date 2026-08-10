"""Table construction and population for strict ground-result evidence."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

from rate_of_closure.simulation.ground_playback import GroundPlaybackTimeline

TRAJECTORY_HEADERS = (
    "Sample",
    "Absolute s",
    "Elapsed s",
    "Phase",
    "x m",
    "y m",
    "z m",
    "vx m/s",
    "vy m/s",
    "vz m/s",
    "ωx rad/s",
    "ωy rad/s",
    "ωz rad/s",
)
EVENT_HEADERS = (
    "Sequence",
    "Event",
    "Time s",
    "x m",
    "y m",
    "z m",
    "vx before m/s",
    "vy before m/s",
    "vz before m/s",
    "vx after m/s",
    "vy after m/s",
    "vz after m/s",
    "ωx before rad/s",
    "ωy before rad/s",
    "ωz before rad/s",
    "ωx after rad/s",
    "ωy after rad/s",
    "ωz after rad/s",
)


def create_ground_table(headers: tuple[str, ...], accessible_name: str) -> QTableWidget:
    """Return a read-only, row-selecting evidence table."""
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setAccessibleName(accessible_name)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    vertical_header = table.verticalHeader()
    if vertical_header is not None:
        vertical_header.setVisible(False)
    return table


def populate_ground_tables(
    timeline: GroundPlaybackTimeline,
    *,
    summary_table: QTableWidget,
    trajectory_table: QTableWidget,
    events_table: QTableWidget,
    warnings_table: QTableWidget,
) -> None:
    """Populate every evidence table from one validated immutable timeline."""
    result = timeline.result
    summary = result.summary
    if summary is None:  # guarded by GroundPlaybackTimeline
        raise RuntimeError("playable result has no summary")
    endpoint = "Total" if timeline.is_complete else "Observed total"
    metrics = (
        ("Carry", summary.carry_distance_m),
        (endpoint, summary.total_distance_m),
        ("Bounce air", summary.bounce_air_distance_m),
        ("Skid", summary.skid_distance_m),
        ("Roll", summary.roll_distance_m),
        ("Surface path", summary.surface_path_distance_m),
    )
    _set_rows(summary_table, [(name, f"{value:.3f} m") for name, value in metrics])
    _set_rows(
        trajectory_table,
        [
            (
                index,
                point.time_s,
                point.time_s - timeline.start_time_s,
                point.phase.value,
                *point.position_m,
                *point.velocity_m_s,
                *point.angular_velocity_rad_s,
            )
            for index, point in enumerate(result.trajectory)
        ],
    )
    _set_rows(
        events_table,
        [
            (
                event.sequence,
                event.event_type.value,
                event.time_s,
                *event.position_m,
                *event.velocity_before_m_s,
                *event.velocity_after_m_s,
                *event.angular_velocity_before_rad_s,
                *event.angular_velocity_after_rad_s,
            )
            for event in result.events
        ],
    )
    warning_rows: list[tuple[object, ...]] = [
        (item.severity.value, item.code, item.message) for item in result.warnings
    ]
    warning_rows.extend(
        [
            ("identity", "schema version", result.schema_version),
            ("identity", "request ID", result.request_id),
            ("identity", "status", result.status.value),
            ("identity", "unit system", result.unit_system),
            ("identity", "frame", result.frame.value),
            ("identity", "surface ID", result.surface_id),
            (
                "identity",
                "model",
                f"{result.model_id} {result.model_version}",
            ),
            (
                "termination",
                result.termination.reason.value,
                f"completed={result.termination.completed}; "
                f"time_s={result.termination.time_s:.6f}",
            ),
            (
                "provenance",
                "producer",
                f"{result.provenance.producer} {result.provenance.producer_version}",
            ),
            ("provenance", "source revision", result.provenance.source_revision),
            ("provenance", "input SHA-256", result.provenance.input_sha256),
            ("calibration", "calibration ID", result.calibration.calibration_id),
            ("calibration", result.calibration.kind.value, result.calibration.source),
            (
                "calibration",
                "confidence",
                f"{result.calibration.confidence:.2f}",
            ),
        ]
    )
    _set_rows(warnings_table, warning_rows)


def _set_rows(table: QTableWidget, rows: Sequence[Sequence[object]]) -> None:
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
            table.setItem(row_index, column, QTableWidgetItem(text))
    table.resizeColumnsToContents()


__all__ = [
    "EVENT_HEADERS",
    "TRAJECTORY_HEADERS",
    "create_ground_table",
    "populate_ground_tables",
]
