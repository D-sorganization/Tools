"""Hash-verified expected-strokes baselines and source-backed SG bookkeeping.

Error posture — ADR-0048 decision G1-D3 (exclude-and-audit)
-----------------------------------------------------------
UpstreamDrift's ``docs/adr/0048-launch-monitor-port-plan.md`` decision G1-D3
rules that "the canonical layer excludes a malformed row, records it against a
``reason_code``, sets ``status='partial'``, and returns a result", that raising
on a malformed row is not canonical behaviour, and that silently dropping a row
is prohibited outright. Its *Consequence* paragraph names this module by name:
``calculate_source_backed_strokes_gained`` "stops raising on out-of-baseline
states, invalid distances, and unknown strata", "and the silent-drop case gains
an exclusion record".

This module implements exactly that. A malformed shot no longer destroys the
session: it is excluded, recorded in :class:`StrokesGainedExcludedRow` against
one of the three canonical ``reason_code`` values, counted in
:class:`StrokesGainedExclusionSummary`, and the result's ``status`` degrades to
``"partial"``. A caller that wants fail-closed behaviour raises on
``status != "available"``; a caller handed an exception could not recover the
good rows.

Reason codes, the exclusion summary, and the three-valued ``status`` mirror the
canonical layer's ``ExcludedRowV1`` / ``ExclusionSummaryV1`` /
``StrokesGainedAnalysisResultV1`` in
``shared.python.launch_monitor.strokes_gained_types``, so the two stacks
classify the same malformed row identically. The TypeScript twin in
``web/src/model/launchMonitorSourceBackedStrokesGained.ts`` carries the same
surface under camelCase names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .launch_monitor_strokes_gained_baseline import (
    CONTRACT_VERSION,
    StrokesGainedBaseline,
    baseline_table_hash,
    load_strokes_gained_baseline,
)

YARDS_PER_METRE = 1.0936132983377078

ExclusionReasonCode = Literal[
    "missing_course_state",
    "invalid_distance",
    "outside_baseline",
]
ResultStatus = Literal["available", "partial", "unavailable"]

EXCLUSION_REASON_CODES: tuple[ExclusionReasonCode, ...] = (
    "missing_course_state",
    "invalid_distance",
    "outside_baseline",
)


class _RowIssue(Exception):
    """One row's disqualifying defect, classified for the audit trail."""

    def __init__(self, reason_code: ExclusionReasonCode, message: str) -> None:
        super().__init__(message)
        self.reason_code: ExclusionReasonCode = reason_code
        self.message = message


@dataclass(frozen=True)
class SourceBackedStrokesGainedRequest:
    """Map retained course-state columns to one verified baseline."""

    before_lie_column: str
    before_context_column: str
    before_target_column: str
    before_distance_column: str
    after_lie_column: str
    after_context_column: str
    after_target_column: str
    after_distance_column: str
    before_distance_unit: str
    after_distance_unit: str
    trusted_summary: TrustedSummaryRequest | None = None


@dataclass(frozen=True)
class TrustedSummaryRequest:
    """Explicit user attestation for grouped and longitudinal SG summaries."""

    player_column: str = ""
    session_column: str = ""
    club_column: str = ""
    order_column: str = ""
    order_unit: str = "session"
    evidence: str = "Explicit user attestation in the Tools scoring UI."


@dataclass(frozen=True)
class StrokesGainedBackingRow:
    """Reproducible state lookup and SG result for one complete shot."""

    source_index: int
    before_lie: str
    before_context: str
    before_target: str
    before_distance_yards: float
    after_lie: str
    after_context: str
    after_target: str
    after_distance_yards: float
    expected_before: float
    expected_after: float
    strokes_gained: float


@dataclass(frozen=True)
class StrokesGainedExcludedRow:
    """One shot the calculation refused, with the reason it was refused.

    Mirrors the canonical ``ExcludedRowV1``. ``source_index`` is the row's
    zero-based position in the supplied frame, so a caller can map an
    exclusion straight back to its input record.
    """

    source_index: int
    reason_code: ExclusionReasonCode
    message: str


@dataclass(frozen=True)
class StrokesGainedExclusionSummary:
    """Row accounting for one calculation. Mirrors ``ExclusionSummaryV1``.

    ``input_row_count == included_row_count + total_excluded`` always holds:
    no row leaves the calculation unaccounted for.
    """

    input_row_count: int
    included_row_count: int
    total_excluded: int
    by_reason: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceBackedStrokesGainedResult:
    """Traceable source-backed SG values, baseline identity, and audit trail.

    ``status`` is ``"available"`` when every supplied row was scored,
    ``"partial"`` when at least one row was scored and at least one excluded,
    and ``"unavailable"`` when no row could be scored. ``mean`` is ``None``
    exactly when ``status == "unavailable"``.
    """

    metric_name: str
    unit: str
    values: tuple[float, ...]
    mean: float | None
    baseline_id: str
    baseline_version: str
    source_url: str
    license: str
    table_sha256: str
    backing_rows: tuple[StrokesGainedBackingRow, ...]
    formula: str
    status: ResultStatus
    excluded_rows: tuple[StrokesGainedExcludedRow, ...]
    exclusions: StrokesGainedExclusionSummary


def _baseline_document(baseline: StrokesGainedBaseline) -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "baseline_id": baseline.baseline_id,
        "version": baseline.version,
        "source_url": baseline.source_url,
        "license": baseline.license,
        "table_sha256": baseline.table_sha256,
        "states": [
            {
                "lie": state.lie,
                "context": state.context,
                "target": state.target,
                "distance_yards": state.distance_yards,
                "expected_strokes": state.expected_strokes,
                "standard_error": state.standard_error,
            }
            for state in baseline.states
        ],
    }


def build_source_backed_strokes_gained_payload(
    frame: pd.DataFrame,
    baseline: StrokesGainedBaseline,
    request: SourceBackedStrokesGainedRequest,
) -> dict[str, object]:
    """Build the canonical Upstream request without performing statistics."""

    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    request_document: dict[str, object] = {
        "start": {
            "lie_column": request.before_lie_column,
            "context_column": request.before_context_column,
            "target_column": request.before_target_column,
            "distance_column": request.before_distance_column,
            "distance_unit": request.before_distance_unit,
        },
        "finish": {
            "lie_column": request.after_lie_column,
            "context_column": request.after_context_column,
            "target_column": request.after_target_column,
            "distance_column": request.after_distance_column,
            "distance_unit": request.after_distance_unit,
        },
        "min_samples": 1,
    }
    if request.trusted_summary is not None:
        request_document.update(_trusted_summary_document(request.trusted_summary))
    return {
        "records": records,
        "baseline": _baseline_document(baseline),
        "request": request_document,
    }


def _trusted_summary_document(spec: TrustedSummaryRequest) -> dict[str, object]:
    columns = (
        ("player", spec.player_column),
        ("session", spec.session_column),
        ("club", spec.club_column),
    )
    summaries = [
        {
            "dimension": dimension,
            "column": column,
            "trust_level": "explicit_user_attested",
            "evidence": spec.evidence,
        }
        for dimension, column in columns
        if column
    ]
    output: dict[str, object] = {"summaries": summaries}
    if spec.player_column and spec.order_column:
        output["longitudinal"] = {
            "order_column": spec.order_column,
            "order_unit": spec.order_unit,
            "group_column": spec.player_column,
            "group_dimension": "player",
            "trust_level": "explicit_user_attested",
            "evidence": spec.evidence,
            "min_samples": 3,
        }
    return output


def _yards(value: object, unit: str, label: str) -> float:
    """Convert one distance cell to yards, or classify why it cannot be.

    Raises :class:`_RowIssue` (row-level, excluded and audited) for cell
    content, and :class:`ValueError` (request-level, still fatal) for a unit
    the request itself declared wrong.
    """

    if unit not in {"yd", "m"}:
        raise ValueError("distance unit must be 'yd' or 'm'")
    if isinstance(value, bool):
        raise _RowIssue("invalid_distance", f"{label} must be numeric")
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise _RowIssue("missing_course_state", f"{label} is missing")
    distance = float(numeric)
    if not np.isfinite(distance) or distance < 0.0:
        raise _RowIssue("invalid_distance", f"{label} must be finite and nonnegative")
    return distance * (YARDS_PER_METRE if unit == "m" else 1.0)


def _course_state_text(value: Any, label: str) -> str:
    """Normalize a course-state label, classifying a blank one as an exclusion."""

    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    normalized = "" if missing else str(value).strip().lower()
    if not normalized:
        raise _RowIssue("missing_course_state", f"{label} is missing")
    return normalized


def _expected(
    baseline: StrokesGainedBaseline,
    lie: str,
    context: str,
    target: str,
    distance: float,
) -> float:
    candidates = sorted(
        (
            state
            for state in baseline.states
            if (state.lie, state.context, state.target) == (lie, context, target)
        ),
        key=lambda state: state.distance_yards,
    )
    if (
        not candidates
        or distance < candidates[0].distance_yards
        or distance > candidates[-1].distance_yards
    ):
        raise _RowIssue(
            "outside_baseline",
            f"course state {lie}/{context}/{target}/{distance:g} yd "
            "is outside the baseline",
        )
    return float(
        np.interp(
            distance,
            [state.distance_yards for state in candidates],
            [state.expected_strokes for state in candidates],
        )
    )


def _backing_row(
    source_index: int,
    row: pd.Series,
    baseline: StrokesGainedBaseline,
    request: SourceBackedStrokesGainedRequest,
) -> StrokesGainedBackingRow:
    """Score one shot, or raise :class:`_RowIssue` naming why it cannot be."""

    before_lie = _course_state_text(row[request.before_lie_column], "start lie")
    before_context = _course_state_text(
        row[request.before_context_column], "start context"
    )
    before_target = _course_state_text(
        row[request.before_target_column], "start target/hole"
    )
    after_lie = _course_state_text(row[request.after_lie_column], "finish lie")
    after_context = _course_state_text(
        row[request.after_context_column], "finish context"
    )
    after_target = _course_state_text(
        row[request.after_target_column], "finish target/hole"
    )
    before_distance = _yards(
        row[request.before_distance_column],
        request.before_distance_unit,
        "start distance",
    )
    after_distance = _yards(
        row[request.after_distance_column],
        request.after_distance_unit,
        "finish distance",
    )
    expected_before = _expected(
        baseline, before_lie, before_context, before_target, before_distance
    )
    expected_after = _expected(
        baseline, after_lie, after_context, after_target, after_distance
    )
    return StrokesGainedBackingRow(
        source_index,
        before_lie,
        before_context,
        before_target,
        before_distance,
        after_lie,
        after_context,
        after_target,
        after_distance,
        expected_before,
        expected_after,
        expected_before - 1.0 - expected_after,
    )


def _partition_rows(
    frame: pd.DataFrame,
    baseline: StrokesGainedBaseline,
    request: SourceBackedStrokesGainedRequest,
) -> tuple[tuple[StrokesGainedBackingRow, ...], tuple[StrokesGainedExcludedRow, ...]]:
    """Split the frame into scored rows and audited exclusions (ADR-0048 G1-D3).

    Postcondition: ``len(scored) + len(excluded) == len(frame)``. No row is
    dropped without a record.
    """

    columns = (
        request.before_lie_column,
        request.before_context_column,
        request.before_target_column,
        request.before_distance_column,
        request.after_lie_column,
        request.after_context_column,
        request.after_target_column,
        request.after_distance_column,
    )
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"columns are unavailable: {missing}")
    rows: list[StrokesGainedBackingRow] = []
    excluded: list[StrokesGainedExcludedRow] = []
    for source_index, (_, row) in enumerate(frame.iterrows()):
        try:
            rows.append(_backing_row(source_index, row, baseline, request))
        except _RowIssue as issue:
            excluded.append(
                StrokesGainedExcludedRow(source_index, issue.reason_code, issue.message)
            )
    return tuple(rows), tuple(excluded)


def calculate_source_backed_strokes_gained(
    frame: pd.DataFrame,
    baseline: StrokesGainedBaseline,
    request: SourceBackedStrokesGainedRequest,
) -> SourceBackedStrokesGainedResult:
    """Calculate SG from verified expected-strokes course-state lookups.

    Per ADR-0048 decision G1-D3 this never raises on row content. A malformed
    shot is excluded, classified against a ``reason_code``, and counted in the
    returned ``exclusions`` summary; ``status`` degrades to ``"partial"`` when
    any row is excluded and to ``"unavailable"`` when none can be scored.
    ``ValueError`` remains reserved for request-level defects the caller
    declared — absent columns, or a distance unit that is not ``yd``/``m``.

    Callers that need fail-closed behaviour check ``status != "available"``.
    """

    rows, excluded = _partition_rows(frame, baseline, request)
    values = tuple(row.strokes_gained for row in rows)
    by_reason: dict[str, int] = {}
    for item in excluded:
        by_reason[item.reason_code] = by_reason.get(item.reason_code, 0) + 1
    if not rows:
        status: ResultStatus = "unavailable"
    elif excluded:
        status = "partial"
    else:
        status = "available"
    return SourceBackedStrokesGainedResult(
        "source_backed_strokes_gained",
        "strokes",
        values,
        float(np.mean(values)) if values else None,
        baseline.baseline_id,
        baseline.version,
        baseline.source_url,
        baseline.license,
        baseline.table_sha256,
        rows,
        "SG = verified E(before course state) - 1 - verified E(after course "
        "state); linear interpolation occurs only within the same exact "
        "lie/context/target stratum.",
        status,
        excluded,
        StrokesGainedExclusionSummary(
            input_row_count=len(frame),
            included_row_count=len(rows),
            total_excluded=len(excluded),
            by_reason=by_reason,
        ),
    )


__all__ = [
    "CONTRACT_VERSION",
    "EXCLUSION_REASON_CODES",
    "ExclusionReasonCode",
    "ResultStatus",
    "SourceBackedStrokesGainedRequest",
    "SourceBackedStrokesGainedResult",
    "StrokesGainedBaseline",
    "StrokesGainedExcludedRow",
    "StrokesGainedExclusionSummary",
    "TrustedSummaryRequest",
    "baseline_table_hash",
    "build_source_backed_strokes_gained_payload",
    "calculate_source_backed_strokes_gained",
    "load_strokes_gained_baseline",
]
