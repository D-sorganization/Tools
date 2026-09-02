"""Canonical source-backed strokes-gained analysis.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/strokes_gained.py`` (432 lines) under
ADR-0046 Stage 1 — step **P14** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``). The implementation is
UpstreamDrift's, carried over rather than reimplemented; its authors retain
authorship.

P14 is one of the two rows the plan marks as carrying a mandated change rather
than being a pure port. Both are recorded here.

Decision G1-D2 — the canonical inference unit is the session cell
------------------------------------------------------------------
This is an *intra-UpstreamDrift* contradiction the port had to settle, not a
cross-stack one. G0 divergence D5 records that this module fit each player's
trend over all 40 shots, treating eight shots from one session as eight
independent observations, while UpstreamDrift's own ``longitudinal.py``
aggregates to 20 player-session cells and its docstring names "player-session
as the inference unit" while warning against exactly that pseudo-replication.
The plan's ruling: "When a repository's two modules disagree and one of them
has already written down why the other is wrong, the written-down argument
wins." ``rate_of_closure.launch_monitor_longitudinal`` independently made the
same choice, so the session cell is also the option that requires no change on
this side.

So ``LongitudinalDimensionV1.method`` now selects the estimand and
``LongitudinalSummaryV1.method`` names it in every result. The canonical
default is ``session-cell-sg-trend/1``; UpstreamDrift's shot-level fit is
preserved unchanged as ``shot-level-sg-trend/1`` and is never reported as the
same quantity. Nothing is removed — the plan's constraint is that no
functionality is deleted or limited, and both estimands remain reachable.

Decision G1-D3 — the canonical error posture is exclude-and-audit
-----------------------------------------------------------------
G1-D3 rules that "the canonical layer excludes a malformed row, records it
against a ``reason_code``, sets ``status='partial'``, and returns a result",
and that raising is not canonical behaviour while silent dropping is
prohibited outright. **This module already is that posture** — see
``_RowIssue``, ``_analyze_rows`` and the ``by_reason`` accounting below — so
the canonical layer satisfies G1-D3 as ported, with no edit.

G1-D3's *Consequence* paragraph additionally requires the legacy
``rate_of_closure.launch_monitor_strokes_gained.calculate_source_backed_strokes_gained``
to stop raising. That change is **not** in this PR, and the reason is
mechanical rather than editorial: it is a cross-repository, cross-runtime
change this PR cannot make atomically. See the PR body for the evidence — in
short, UpstreamDrift's G0 gate pins ``SourceBackedStrokesGainedResult``'s
dataclass field set *exactly* (D2) and pins the raise itself (D1), the result
has a TypeScript twin with pinned cross-runtime goldens, and the gate file
lives in UpstreamDrift where a Tools PR cannot re-pin it.

The baseline argument is structural
------------------------------------
``baseline`` is typed
:class:`~shared.python.launch_monitor.strokes_gained_types.ExpectedStrokesBaselineLike`
rather than a pydantic model, because P12 left the expected-strokes baseline
half behind: ``rate_of_closure.launch_monitor_strokes_gained_baseline`` is the
already-home authority for loading, byte-capping, URL-validating and digest-
verifying that artifact, and G0 pinned the two digests as identical. Typing
structurally lets the already-home ``StrokesGainedBaseline`` flow straight in
without this package importing ``rate_of_closure``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
from typing import Literal

import pandas as pd

from shared.python.launch_monitor._scoring_statistics import (
    estimate_summary,
    group_summaries,
    longitudinal_summaries,
)
from shared.python.launch_monitor.contract_v2 import (
    AnalysisContextV2,
    _record_digest,
)
from shared.python.launch_monitor.strokes_gained_types import (
    BASELINE_CONTRACT_VERSION,
    AvailabilityV1,
    BaselineProvenanceV1,
    CourseStateColumnsV1,
    CourseStateValueV1,
    EstimateSummaryV1,
    ExcludedRowV1,
    ExclusionSummaryV1,
    ExpectedStrokesBaselineLike,
    ExpectedStrokesStateLike,
    InterpolationV1,
    StrokesGainedAnalysisResultV1,
    StrokesGainedRequestV1,
    StrokesGainedRowV1,
    StrokesGainedUncertaintyV1,
)

YARDS_PER_METRE = 1.0936132983377078


@dataclass(frozen=True)
class _Lookup:
    expected: float
    standard_error: float | None
    interpolation: InterpolationV1


class _RowIssue(ValueError):
    def __init__(
        self,
        reason_code: Literal[
            "missing_course_state", "invalid_distance", "outside_baseline"
        ],
        message: str,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _required_columns(request: StrokesGainedRequestV1) -> set[str]:
    columns = {
        request.start.lie_column,
        request.start.context_column,
        request.start.target_column,
        request.start.distance_column,
        request.finish.lie_column,
        request.finish.context_column,
        request.finish.target_column,
        request.finish.distance_column,
    }
    columns.update(summary.column for summary in request.summaries)
    if request.shot_id_column:
        columns.add(request.shot_id_column)
    if request.longitudinal:
        columns.add(request.longitudinal.order_column)
        if request.longitudinal.group_column:
            columns.add(request.longitudinal.group_column)
    return columns


def _validate_columns(frame: pd.DataFrame, request: StrokesGainedRequestV1) -> None:
    missing = sorted(_required_columns(request).difference(frame.columns))
    if missing:
        raise ValueError(f"Columns not present in launch-monitor records: {missing}")


def _text(value: object, label: str) -> str:
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    normalized = "" if missing else str(value).strip().lower()
    if not normalized:
        raise _RowIssue("missing_course_state", f"{label} is missing")
    return normalized


def _yards(value: object, unit: str, label: str) -> float:
    if isinstance(value, bool):
        raise _RowIssue("invalid_distance", f"{label} must be numeric")
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise _RowIssue("missing_course_state", f"{label} is missing")
    distance = float(numeric)
    if not isfinite(distance) or distance < 0:
        raise _RowIssue("invalid_distance", f"{label} must be finite and nonnegative")
    return distance * (YARDS_PER_METRE if unit == "m" else 1.0)


def _course_state(
    row: dict,
    columns: CourseStateColumnsV1,
    label: str,
) -> CourseStateValueV1:
    return CourseStateValueV1(
        lie=_text(row[columns.lie_column], f"{label} lie"),
        context=_text(row[columns.context_column], f"{label} context"),
        target=_text(row[columns.target_column], f"{label} target/hole"),
        distance_yards=_yards(
            row[columns.distance_column], columns.distance_unit, f"{label} distance"
        ),
    )


def _interpolated_error(
    lower: ExpectedStrokesStateLike,
    upper: ExpectedStrokesStateLike,
    fraction: float,
) -> float | None:
    if lower.standard_error is None or upper.standard_error is None:
        return None
    return lower.standard_error + fraction * (
        upper.standard_error - lower.standard_error
    )


def _lookup(
    baseline: ExpectedStrokesBaselineLike,
    state: CourseStateValueV1,
) -> _Lookup:
    matches = sorted(
        (
            point
            for point in baseline.states
            if (
                point.lie,
                point.context,
                point.target,
            )
            == (state.lie, state.context, state.target)
        ),
        key=lambda point: point.distance_yards,
    )
    if (
        not matches
        or state.distance_yards < matches[0].distance_yards
        or state.distance_yards > matches[-1].distance_yards
    ):
        raise _RowIssue(
            "outside_baseline",
            "course state is absent from or outside the benchmark range",
        )
    upper_index = next(
        index
        for index, point in enumerate(matches)
        if point.distance_yards >= state.distance_yards
    )
    upper = matches[upper_index]
    lower = upper if upper_index == 0 else matches[upper_index - 1]
    span = upper.distance_yards - lower.distance_yards
    fraction = (
        0.0 if span == 0 else (state.distance_yards - lower.distance_yards) / span
    )
    expected = lower.expected_strokes + fraction * (
        upper.expected_strokes - lower.expected_strokes
    )
    return _Lookup(
        expected=float(expected),
        standard_error=_interpolated_error(lower, upper, fraction),
        interpolation=InterpolationV1(
            lower_distance_yards=lower.distance_yards,
            upper_distance_yards=upper.distance_yards,
            fraction=float(fraction),
        ),
    )


def _optional_id(row: dict, column: str | None) -> str | None:
    if not column:
        return None
    value = row[column]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip()
    return normalized or None


def _groups(row: dict, request: StrokesGainedRequestV1) -> dict[str, str]:
    output: dict[str, str] = {}
    for summary in request.summaries:
        value = _optional_id(row, summary.column)
        if value:
            output[summary.dimension] = value
    longitudinal = request.longitudinal
    if longitudinal and longitudinal.group_column and longitudinal.group_dimension:
        value = _optional_id(row, longitudinal.group_column)
        if value:
            output[longitudinal.group_dimension] = value
    return output


def _order(row: dict, request: StrokesGainedRequestV1) -> float | None:
    if request.longitudinal is None:
        return None
    numeric = pd.to_numeric(
        pd.Series([row[request.longitudinal.order_column]]), errors="coerce"
    ).iloc[0]
    if pd.isna(numeric) or not isfinite(float(numeric)):
        return None
    return float(numeric)


def _row_result(
    source_index: int,
    raw_row: dict,
    baseline: ExpectedStrokesBaselineLike,
    request: StrokesGainedRequestV1,
) -> StrokesGainedRowV1:
    start = _course_state(raw_row, request.start, "start")
    finish = _course_state(raw_row, request.finish, "finish")
    expected_start = _lookup(baseline, start)
    expected_finish = _lookup(baseline, finish)
    benchmark_error = None
    if (
        expected_start.standard_error is not None
        and expected_finish.standard_error is not None
    ):
        benchmark_error = sqrt(
            expected_start.standard_error**2 + expected_finish.standard_error**2
        )
    raw = {str(key): value for key, value in raw_row.items()}
    return StrokesGainedRowV1(
        source_index=source_index,
        shot_id=_optional_id(raw_row, request.shot_id_column),
        input_record_sha256=_record_digest(raw),
        start=start,
        finish=finish,
        expected_start=expected_start.expected,
        expected_finish=expected_finish.expected,
        benchmark_standard_error=benchmark_error,
        strokes_gained=expected_start.expected - 1.0 - expected_finish.expected,
        start_interpolation=expected_start.interpolation,
        finish_interpolation=expected_finish.interpolation,
        groups=_groups(raw_row, request),
        longitudinal_order=_order(raw_row, request),
    )


def _analyze_rows(
    frame: pd.DataFrame,
    baseline: ExpectedStrokesBaselineLike,
    request: StrokesGainedRequestV1,
) -> tuple[tuple[StrokesGainedRowV1, ...], tuple[ExcludedRowV1, ...]]:
    included: list[StrokesGainedRowV1] = []
    excluded: list[ExcludedRowV1] = []
    # ⚡ Bolt: Vectorized dictionary conversion is ~700x faster than iterrows()
    records = frame.to_dict("records")
    for source_index, raw_row in enumerate(records):
        try:
            included.append(_row_result(source_index, raw_row, baseline, request))
        except _RowIssue as error:
            excluded.append(
                ExcludedRowV1(
                    source_index=source_index,
                    shot_id=_optional_id(raw_row, request.shot_id_column),
                    reason_code=error.reason_code,
                    message=str(error),
                )
            )
    return tuple(included), tuple(excluded)


def _availability(
    count: int, required: int
) -> tuple[Literal["available", "unavailable"], AvailabilityV1]:
    if count >= required:
        return "available", AvailabilityV1(
            state="available", observed_count=count, required_count=required
        )
    return "unavailable", AvailabilityV1(
        state="unavailable",
        reason_code="insufficient_complete_rows",
        message="Too few complete, benchmark-covered course-state rows.",
        observed_count=count,
        required_count=required,
    )


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    # ⚡ Bolt: Vectorized dictionary conversion is ~700x faster than iterrows()
    digests = [
        _record_digest({str(key): value for key, value in row.items()})
        for row in frame.to_dict("records")
    ]
    return sha256("\n".join(digests).encode("ascii")).hexdigest()


def _uncertainty(
    rows: tuple[StrokesGainedRowV1, ...], level: float
) -> StrokesGainedUncertaintyV1:
    errors = [row.benchmark_standard_error for row in rows]
    complete = [value for value in errors if value is not None]
    benchmark_mean_error = (
        sqrt(sum(value**2 for value in complete)) / len(complete)
        if complete and len(complete) == len(rows)
        else None
    )
    return StrokesGainedUncertaintyV1(
        sampling_method="student-t-descriptive-mean",
        confidence_level=level,
        benchmark_method=(
            "interpolated-state-standard-errors"
            if benchmark_mean_error is not None
            else "unavailable"
        ),
        benchmark_standard_error_mean=benchmark_mean_error,
        assumptions=(
            "The sampling interval treats included shots as independent.",
            "Benchmark state errors are combined as independent when supplied.",
            "Interpolation stays within an exact lie/context/target stratum.",
        ),
    )


def _baseline_provenance(baseline: ExpectedStrokesBaselineLike) -> BaselineProvenanceV1:
    """Record which baseline artifact produced a result.

    ``contract_version`` is read structurally: the already-home
    ``StrokesGainedBaseline`` publishes it as its module-level
    ``CONTRACT_VERSION`` rather than as a field, and both stacks spell the
    value identically.
    """

    return BaselineProvenanceV1(
        baseline_id=baseline.baseline_id,
        version=baseline.version,
        source_url=baseline.source_url,
        license=baseline.license,
        table_sha256=baseline.table_sha256,
        contract_version=str(
            getattr(baseline, "contract_version", BASELINE_CONTRACT_VERSION)
        ),
    )


def analyze_source_backed_strokes_gained(
    frame: pd.DataFrame,
    baseline: ExpectedStrokesBaselineLike,
    request: StrokesGainedRequestV1,
    *,
    context: AnalysisContextV2 | None = None,
) -> StrokesGainedAnalysisResultV1:
    """Return governed SG values without mutating caller records.

    Postcondition: every reported value links to a hash-verified benchmark and
    an explicit start and finish lie/context/target/distance state.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    _validate_columns(frame, request)
    rows, excluded = _analyze_rows(frame, baseline, request)
    availability_state, availability = _availability(len(rows), request.min_samples)
    status: Literal["available", "partial", "unavailable"] = availability_state
    if status == "available" and excluded:
        status = "partial"
    by_reason: dict[str, int] = {}
    for row in excluded:
        by_reason[row.reason_code] = by_reason.get(row.reason_code, 0) + 1
    warnings = (
        (f"{len(excluded)} rows were excluded under declared rules.",)
        if excluded
        else ()
    )
    summary = (
        estimate_summary([row.strokes_gained for row in rows], request.confidence_level)
        if availability_state == "available"
        else EstimateSummaryV1(count=len(rows))
    )
    grouped = (
        group_summaries(rows, request.summaries, request.confidence_level)
        if availability_state == "available"
        else ()
    )
    longitudinal = (
        longitudinal_summaries(rows, request.longitudinal)
        if availability_state == "available"
        else ()
    )
    return StrokesGainedAnalysisResultV1(
        status=status,
        value_summary=summary,
        baseline=_baseline_provenance(baseline),
        formula=(
            "SG = verified E(start lie/context/target/distance) - 1 - "
            "verified E(finish lie/context/target/distance)"
        ),
        units={"strokes_gained": "strokes", "distance": "yd"},
        availability=availability,
        uncertainty=_uncertainty(rows, request.confidence_level),
        row_results=rows,
        excluded_rows=excluded,
        exclusions=ExclusionSummaryV1(
            input_row_count=len(frame),
            included_row_count=len(rows),
            total_excluded=len(excluded),
            by_reason=by_reason,
        ),
        group_summaries=grouped,
        longitudinal_summaries=longitudinal,
        analysis_context=context or AnalysisContextV2(),
        dataset_fingerprint_sha256=_dataset_fingerprint(frame),
        warnings=warnings,
        limitations=(
            "This is descriptive scoring bookkeeping, not causal inference.",
            "Target/hole, lie, and context labels must be supplied and valid.",
            "The baseline declaration is not an independent license audit.",
            "Results outside benchmark support fail closed rather than extrapolate.",
        ),
    )


def strokes_gained_contract_json_schema() -> dict[str, object]:
    """Return the canonical result schema published to static clients."""

    return StrokesGainedAnalysisResultV1.model_json_schema()


__all__ = [
    "YARDS_PER_METRE",
    "analyze_source_backed_strokes_gained",
    "strokes_gained_contract_json_schema",
]
