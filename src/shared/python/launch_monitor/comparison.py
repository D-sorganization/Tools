"""Matched and descriptive cross-monitor comparison.

Ported from UpstreamDrift ``src/shared/python/launch_monitor/comparison.py``
(147 lines) under ADR-0046 Stage 1 — step **P4** of the ADR-0046 G1 port plan
(UpstreamDrift ``docs/adr/0048-launch-monitor-port-plan.md``). The
implementation is UpstreamDrift's, carried over unchanged rather than
reimplemented; its authors retain authorship. No behaviour is added, removed,
or limited by the move.

Monitor comparison is one of the three capabilities the port plan's
*Corrections to ADR-0046* confirms is genuinely **UpstreamDrift-only**: a
search of ``rate_of_closure`` for the identifying symbols
(``compare_monitors``, Bland-Altman limits of agreement, Deming/OLS slope
between two monitors) returns nothing. Nothing here collides by name with that
package, and no ADR-0046 G0 divergence applies.

Two comparison modes, and the distinction is the point of the module:

* **matched** (``match_column`` given) — the two monitors measured the *same
  shot*, so the pair is differenced within a match key. This yields a true
  Bland-Altman bias with 95% limits of agreement (``mean_bias`` ±1.96·SD) plus
  the OLS slope, intercept, and correlation of comparator on reference. Three
  matched pairs are the floor; below that a regression through two points is
  not evidence.
* **unmatched** (no ``match_column``) — the two monitors measured *different*
  shots, so only the distributions can be compared. Everything a matched run
  reports about agreement is returned as ``nan``, the ``correlation`` slot
  instead carries a pooled-SD standardised effect size, and the result carries
  an explicit ``warning`` that the difference may be confounded by player,
  club, environment, and session composition. That warning is load-bearing:
  it is what stops a descriptive gap being read as a device bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "MonitorComparisonResult",
    "MonitorSummary",
    "PairwiseMonitorComparison",
    "compare_monitors",
]


@dataclass(frozen=True)
class MonitorSummary:
    monitor: str
    sample_count: int
    mean: float
    standard_deviation: float
    median: float


@dataclass(frozen=True)
class PairwiseMonitorComparison:
    reference: str
    comparator: str
    matched: bool
    sample_count: int
    mean_bias: float
    standard_deviation_bias: float
    lower_limit: float
    upper_limit: float
    slope: float
    intercept: float
    correlation: float
    warning: str | None


@dataclass(frozen=True)
class MonitorComparisonResult:
    metric: str
    summaries: tuple[MonitorSummary, ...]
    pairwise: tuple[PairwiseMonitorComparison, ...]


def compare_monitors(
    frame: pd.DataFrame,
    *,
    metric: str,
    monitor_column: str = "monitor_vendor",
    match_column: str | None = None,
    reference_monitor: str | None = None,
) -> MonitorComparisonResult:
    """Compare monitor distributions or matched-shot measurement behavior."""
    required = {metric, monitor_column}
    if match_column:
        required.add(match_column)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Columns not present: {sorted(missing)}")
    clean = frame[list(required)].copy()
    clean[metric] = pd.to_numeric(clean[metric], errors="coerce")
    clean = clean.dropna(subset=[metric, monitor_column])
    monitors = sorted(clean[monitor_column].astype(str).unique())
    if len(monitors) < 2:
        raise ValueError("At least two monitors are required")
    reference = reference_monitor or monitors[0]
    if reference not in monitors:
        raise ValueError(f"Reference monitor not present: {reference}")
    summaries = tuple(
        MonitorSummary(
            monitor,
            int(len(values)),
            float(values.mean()),
            float(values.std(ddof=1)),
            float(values.median()),
        )
        for monitor in monitors
        for values in [clean.loc[clean[monitor_column].astype(str) == monitor, metric]]
    )
    pairwise: list[PairwiseMonitorComparison] = []
    for comparator in monitors:
        if comparator == reference:
            continue
        if match_column:
            selected = clean[
                clean[monitor_column].astype(str).isin([reference, comparator])
            ]
            pivot = selected.pivot_table(
                index=match_column,
                columns=monitor_column,
                values=metric,
                aggfunc="mean",
            ).dropna(subset=[reference, comparator])
            if len(pivot) < 3:
                raise ValueError(
                    "At least three matched pairs are required for "
                    f"{reference} vs {comparator}"
                )
            x = pivot[reference].to_numpy(float)
            y = pivot[comparator].to_numpy(float)
            difference = y - x
            slope, intercept = np.polyfit(x, y, 1)
            bias = float(difference.mean())
            std_bias = float(difference.std(ddof=1))
            pairwise.append(
                PairwiseMonitorComparison(
                    reference,
                    comparator,
                    True,
                    len(pivot),
                    bias,
                    std_bias,
                    bias - 1.96 * std_bias,
                    bias + 1.96 * std_bias,
                    float(slope),
                    float(intercept),
                    float(np.corrcoef(x, y)[0, 1]),
                    None,
                )
            )
        else:
            reference_values = clean.loc[
                clean[monitor_column].astype(str) == reference, metric
            ].to_numpy(float)
            comparator_values = clean.loc[
                clean[monitor_column].astype(str) == comparator, metric
            ].to_numpy(float)
            bias = float(comparator_values.mean() - reference_values.mean())
            pooled = np.sqrt(
                (reference_values.var(ddof=1) + comparator_values.var(ddof=1)) / 2
            )
            effect = bias / pooled if pooled > 0 else float("nan")
            pairwise.append(
                PairwiseMonitorComparison(
                    reference,
                    comparator,
                    False,
                    min(len(reference_values), len(comparator_values)),
                    bias,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float(effect),
                    "Unmatched comparison is descriptive and may be confounded by "
                    "player, club, environment, and session composition.",
                )
            )
    return MonitorComparisonResult(metric, summaries, tuple(pairwise))
