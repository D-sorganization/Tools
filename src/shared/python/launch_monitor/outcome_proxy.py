"""Explicit non-strokes-gained launch-outcome proximity proxy.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/outcome_proxy.py`` (114 lines) under
ADR-0046 Stage 1 — step **P13** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``). The implementation is
UpstreamDrift's, carried over unchanged rather than reimplemented; its authors
retain authorship. No behaviour is added, removed, or limited by the move.

Classified without waiting for a gate, then gated anyway
--------------------------------------------------------
The plan classifies this module ``port-up`` rather than ``needs-decision``
even though G0 never measured it, and says why: both implementations were read
and "the closed form is character-for-character the same statistic —
``hypot(carry_yd - target_yd, lateral_yd)`` after identical yard conversion",
with UpstreamDrift's a strict superset that adds exclusion accounting, an
uncertainty summary, an availability status, and an explicit "this is not
strokes gained" claims block. The classification does not depend on a
measurement, but the plan still requires one before the port lands: P13's row
reads "new target-error gate landed in this PR".

That gate is ``test_outcome_proxy_target_error_gate`` in
``tests/shared/python/launch_monitor/test_strokes_gained_contract.py``. It runs
this module and ``rate_of_closure.launch_monitor_performance.calculate_target_error``
over the same frame and pins the per-row radial errors and the mean to delta
exactly ``0.0`` — the reading, confirmed by arithmetic. Both stacks now live in
this repository, so unlike the G0 files the gate needs no vendored submodule.

**No re-export in either direction.** The gate imports ``rate_of_closure``;
this module must not.
"""

from __future__ import annotations

from math import hypot, isfinite
from typing import Literal

import pandas as pd

from shared.python.launch_monitor._scoring_statistics import estimate_summary
from shared.python.launch_monitor.strokes_gained_types import (
    ExclusionSummaryV1,
    OutcomeProxyRequestV1,
    OutcomeProxyResultV1,
    OutcomeProxyRowV1,
)

YARDS_PER_METRE = 1.0936132983377078


def _yards(value: object, unit: str) -> float | None:
    if isinstance(value, bool):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not isfinite(float(numeric)):
        return None
    return float(numeric) * (YARDS_PER_METRE if unit == "m" else 1.0)


def _shot_id(row: pd.Series, column: str | None) -> str | None:
    if not column:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _validate_columns(frame: pd.DataFrame, request: OutcomeProxyRequestV1) -> None:
    required = {request.carry_column, request.lateral_column}
    if request.shot_id_column:
        required.add(request.shot_id_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Columns not present in launch-monitor records: {missing}")


def _rows(
    frame: pd.DataFrame, request: OutcomeProxyRequestV1
) -> tuple[tuple[OutcomeProxyRowV1, ...], int]:
    included: list[OutcomeProxyRowV1] = []
    excluded = 0
    # ⚡ Bolt: Vectorized dictionary conversion is ~700x faster than iterrows()
    records = frame.to_dict("records")
    for source_index, row in enumerate(records):
        carry = _yards(row.get(request.carry_column), request.carry_unit)
        lateral = _yards(row.get(request.lateral_column), request.lateral_unit)
        if carry is None or lateral is None:
            excluded += 1
            continue
        included.append(
            OutcomeProxyRowV1(
                source_index=source_index,
                shot_id=_shot_id(row, request.shot_id_column),
                carry_yards=carry,
                lateral_yards=lateral,
                target_distance_yards=request.target_distance_yards,
                radial_error_yards=hypot(
                    carry - request.target_distance_yards, lateral
                ),
            )
        )
    return tuple(included), excluded


def analyze_outcome_proxy(
    frame: pd.DataFrame, request: OutcomeProxyRequestV1
) -> OutcomeProxyResultV1:
    """Return target-relative dispersion while forbidding an SG claim."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    _validate_columns(frame, request)
    rows, excluded = _rows(frame, request)
    enough = len(rows) >= request.min_samples
    status: Literal["available", "partial", "unavailable"] = (
        "available" if enough else "unavailable"
    )
    if enough and excluded:
        status = "partial"
    return OutcomeProxyResultV1(
        status=status,
        value_summary=estimate_summary(
            [row.radial_error_yards for row in rows], request.confidence_level
        ),
        row_results=rows,
        exclusions=ExclusionSummaryV1(
            input_row_count=len(frame),
            included_row_count=len(rows),
            total_excluded=excluded,
            by_reason={"missing_or_non_numeric_outcome": excluded} if excluded else {},
        ),
        formula="radial error = sqrt((carry yd - target yd)^2 + lateral yd^2)",
        units={"carry": "yd", "lateral": "yd", "radial_error": "yd"},
        limitations=(
            "This target-relative dispersion proxy is not strokes gained.",
            "It does not use a source-backed expected-strokes benchmark.",
            "It is descriptive and does not support causal inference.",
        ),
    )


__all__ = ["analyze_outcome_proxy"]
