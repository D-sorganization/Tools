"""Hash-verified expected-strokes baselines and source-backed SG bookkeeping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .launch_monitor_strokes_gained_baseline import (
    CONTRACT_VERSION,
    StrokesGainedBaseline,
    baseline_table_hash,
    load_strokes_gained_baseline,
)

YARDS_PER_METRE = 1.0936132983377078


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
class SourceBackedStrokesGainedResult:
    """Traceable source-backed SG values and baseline identity."""

    metric_name: str
    unit: str
    values: tuple[float, ...]
    mean: float
    baseline_id: str
    baseline_version: str
    source_url: str
    license: str
    table_sha256: str
    backing_rows: tuple[StrokesGainedBackingRow, ...]
    formula: str


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


def _yards(value: object, unit: str) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(numeric):
        return None
    if unit == "yd":
        return float(numeric)
    if unit == "m":
        return float(numeric) * YARDS_PER_METRE
    raise ValueError("distance unit must be 'yd' or 'm'")


def _course_state_text(value: Any) -> str:
    """Normalize an optional course-state label without propagating pandas Any."""

    return "" if pd.isna(value) else str(value).strip().lower()


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
        raise ValueError(
            f"course state {lie}/{context}/{target}/{distance:g} yd "
            "is outside the baseline"
        )
    return float(
        np.interp(
            distance,
            [state.distance_yards for state in candidates],
            [state.expected_strokes for state in candidates],
        )
    )


def _backing_rows(
    frame: pd.DataFrame,
    baseline: StrokesGainedBaseline,
    request: SourceBackedStrokesGainedRequest,
) -> tuple[StrokesGainedBackingRow, ...]:
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
    for source_index, (_, row) in enumerate(frame.iterrows()):
        before_value = row[request.before_lie_column]
        before_context_value = row[request.before_context_column]
        before_target_value = row[request.before_target_column]
        after_value = row[request.after_lie_column]
        after_context_value = row[request.after_context_column]
        after_target_value = row[request.after_target_column]
        before_lie = _course_state_text(before_value)
        before_context = _course_state_text(before_context_value)
        before_target = _course_state_text(before_target_value)
        after_lie = _course_state_text(after_value)
        after_context = _course_state_text(after_context_value)
        after_target = _course_state_text(after_target_value)
        before_distance = _yards(
            row[request.before_distance_column], request.before_distance_unit
        )
        after_distance = _yards(
            row[request.after_distance_column], request.after_distance_unit
        )
        if (
            not before_lie
            or not before_context
            or not before_target
            or not after_lie
            or not after_context
            or not after_target
            or before_distance is None
            or after_distance is None
        ):
            continue
        expected_before = _expected(
            baseline,
            before_lie,
            before_context,
            before_target,
            before_distance,
        )
        expected_after = _expected(
            baseline,
            after_lie,
            after_context,
            after_target,
            after_distance,
        )
        gained = expected_before - 1.0 - expected_after
        rows.append(
            StrokesGainedBackingRow(
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
                gained,
            )
        )
    return tuple(rows)


def calculate_source_backed_strokes_gained(
    frame: pd.DataFrame,
    baseline: StrokesGainedBaseline,
    request: SourceBackedStrokesGainedRequest,
) -> SourceBackedStrokesGainedResult:
    """Calculate SG from verified expected-strokes course-state lookups."""

    rows = _backing_rows(frame, baseline, request)
    if not rows:
        raise ValueError(
            "source-backed strokes gained requires complete course-state rows"
        )
    values = tuple(row.strokes_gained for row in rows)
    return SourceBackedStrokesGainedResult(
        "source_backed_strokes_gained",
        "strokes",
        values,
        float(np.mean(values)),
        baseline.baseline_id,
        baseline.version,
        baseline.source_url,
        baseline.license,
        baseline.table_sha256,
        rows,
        "SG = verified E(before course state) - 1 - verified E(after course "
        "state); linear interpolation occurs only within the same exact "
        "lie/context/target stratum.",
    )


__all__ = [
    "CONTRACT_VERSION",
    "SourceBackedStrokesGainedRequest",
    "SourceBackedStrokesGainedResult",
    "StrokesGainedBaseline",
    "TrustedSummaryRequest",
    "baseline_table_hash",
    "build_source_backed_strokes_gained_payload",
    "calculate_source_backed_strokes_gained",
    "load_strokes_gained_baseline",
]
