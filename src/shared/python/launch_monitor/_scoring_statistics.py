"""Shared descriptive uncertainty and trusted grouping helpers.

Ported from UpstreamDrift
``src/shared/python/launch_monitor/_scoring_statistics.py`` (127 lines) under
ADR-0046 Stage 1 — step **P12** of the ADR-0046 G1 port plan (UpstreamDrift
``docs/adr/0048-launch-monitor-port-plan.md``). The implementation is
UpstreamDrift's, carried over rather than reimplemented; its authors retain
authorship.

The port plan's third correction to ADR-0046 is about this file: the ADR's
module list omits it entirely, yet it is "the actual implementation behind
divergences D2, D3, and D4" and "the highest value-per-line module in the
``port-up`` set". D2 is the strokes-gained uncertainty ``rate_of_closure`` has
no field for, D3 the grouped estimates it cannot compute locally, D4 the
longitudinal trend its strokes-gained module does not fit. All three live in
the three public functions below.

Decision G1-D2 lands in :func:`longitudinal_summaries`
------------------------------------------------------
Step **P14**'s row mandates "D5 per G1-D2", and the plan's ratified decision is
that "the canonical estimand for any longitudinal fit is the **player-session
cell**: shots aggregate to one value per player per session before any slope is
fitted." UpstreamDrift's shot-level fit is preserved as the named variant
``shot-level-sg-trend/1`` — never removed, never reported as the same quantity.
See :meth:`_trend` and the module note in
:mod:`shared.python.launch_monitor.strokes_gained`.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy import stats

from shared.python.launch_monitor.strokes_gained_types import (
    ConfidenceIntervalV1,
    EstimateSummaryV1,
    GroupingDimensionV1,
    GroupSummaryV1,
    LongitudinalDimensionV1,
    LongitudinalSummaryV1,
    StrokesGainedRowV1,
)


def estimate_summary(values: list[float], confidence_level: float) -> EstimateSummaryV1:
    """Return a finite descriptive mean and Student-t interval when possible."""

    if not values:
        return EstimateSummaryV1(count=0)
    vector = np.asarray(values, dtype=float)
    mean = float(np.mean(vector))
    if len(vector) == 1:
        return EstimateSummaryV1(count=1, mean=mean)
    deviation = float(np.std(vector, ddof=1))
    standard_error = deviation / sqrt(len(vector))
    critical = float(stats.t.ppf(0.5 + confidence_level / 2.0, len(vector) - 1))
    return EstimateSummaryV1(
        count=len(vector),
        mean=mean,
        standard_deviation=deviation,
        standard_error=standard_error,
        confidence_interval=ConfidenceIntervalV1(
            lower=mean - critical * standard_error,
            upper=mean + critical * standard_error,
            level=confidence_level,
            method="student-t",
        ),
    )


def group_summaries(
    rows: tuple[StrokesGainedRowV1, ...],
    specs: tuple[GroupingDimensionV1, ...],
    confidence_level: float,
) -> tuple[GroupSummaryV1, ...]:
    """Summarize only groups explicitly mapped with evidence in the request."""

    output: list[GroupSummaryV1] = []
    for spec in specs:
        values_by_group: dict[str, list[float]] = {}
        for row in rows:
            value = row.groups.get(spec.dimension)
            if value:
                values_by_group.setdefault(value, []).append(row.strokes_gained)
        for value in sorted(values_by_group):
            output.append(
                GroupSummaryV1(
                    dimension=spec.dimension,
                    group_value=value,
                    estimate=estimate_summary(values_by_group[value], confidence_level),
                    trust_level=spec.trust_level,
                    evidence=spec.evidence,
                )
            )
    return tuple(output)


def _inference_units(
    complete: list[StrokesGainedRowV1],
    spec: LongitudinalDimensionV1,
) -> tuple[list[float], list[float]]:
    """Return the (order, value) pairs the requested estimand actually fits.

    G1-D2: the canonical unit is the session cell, so shots sharing an order
    value collapse to one equal-weight mean before the slope is fitted. The
    ``shot-level-sg-trend/1`` variant keeps UpstreamDrift's original behaviour,
    where every shot is its own observation.
    """

    if spec.method == "shot-level-sg-trend/1":
        return (
            [float(row.longitudinal_order or 0.0) for row in complete],
            [row.strokes_gained for row in complete],
        )
    cells: dict[float, list[float]] = {}
    for row in complete:
        cells.setdefault(float(row.longitudinal_order or 0.0), []).append(
            row.strokes_gained
        )
    order = sorted(cells)
    return order, [float(np.mean(cells[value])) for value in order]


def _trend(
    rows: list[StrokesGainedRowV1],
    spec: LongitudinalDimensionV1,
    group_dimension: str,
    group_value: str,
) -> LongitudinalSummaryV1 | None:
    complete = [row for row in rows if row.longitudinal_order is not None]
    order_values, metric_values = _inference_units(complete, spec)
    if len(order_values) < spec.min_samples:
        return None
    order = np.asarray(order_values, dtype=float)
    if len(np.unique(order)) < 2:
        return None
    values = np.asarray(metric_values, dtype=float)
    estimate = stats.linregress(order, values)
    return LongitudinalSummaryV1(
        group_dimension=group_dimension,  # type: ignore[arg-type]
        group_value=group_value,
        method=spec.method,
        sample_count=len(order_values),
        slope=float(estimate.slope),
        intercept=float(estimate.intercept),
        r_squared=float(estimate.rvalue**2),
        p_value=float(estimate.pvalue),
        slope_unit=f"strokes/{spec.order_unit}",
        trust_level=spec.trust_level,
        evidence=spec.evidence,
    )


def longitudinal_summaries(
    rows: tuple[StrokesGainedRowV1, ...],
    spec: LongitudinalDimensionV1 | None,
) -> tuple[LongitudinalSummaryV1, ...]:
    """Fit descriptive SG trends only for explicitly evidenced order fields."""

    if spec is None:
        return ()
    if spec.group_dimension is None:
        summary = _trend(list(rows), spec, "all", "all")
        return () if summary is None else (summary,)
    grouped: dict[str, list[StrokesGainedRowV1]] = {}
    for row in rows:
        value = row.groups.get(spec.group_dimension)
        if value:
            grouped.setdefault(value, []).append(row)
    output = [
        summary
        for value in sorted(grouped)
        if (summary := _trend(grouped[value], spec, spec.group_dimension, value))
        is not None
    ]
    return tuple(output)


__all__ = ["estimate_summary", "group_summaries", "longitudinal_summaries"]
