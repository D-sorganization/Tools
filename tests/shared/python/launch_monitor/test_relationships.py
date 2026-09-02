"""Canonical relationship-analysis tests (ADR-0046 G1 step P7).

The first case is UpstreamDrift's
``tests/unit/launch_monitor/test_analysis.py::test_correlations_include_counts_significance_and_derived_warning``,
travelling verbatim with the module it exercises. The remaining cases pin the
module's refusals and — deliberately — the two behaviours that owner rulings
**D15** and **D17** will change in a follow-up PR. This port carries
UpstreamDrift's behaviour unchanged; pinning today's behaviour here is what
makes the follow-up's diff visible instead of invisible.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from shared.python.launch_monitor.relationships import compute_correlations

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_METRICS = ("club_speed", "ball_speed", "smash_factor", "attack_angle")


def test_correlations_include_counts_significance_and_derived_warning(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Ported verbatim from UpstreamDrift's ``test_analysis.py``."""
    result = compute_correlations(
        shots(),
        metrics=_METRICS,
        method="spearman",
        controls=("attack_angle",),
    )
    assert result.coefficients.loc["club_speed", "ball_speed"] > 0.95
    assert result.p_values.loc["club_speed", "ball_speed"] < 0.001
    assert result.pair_counts.loc["club_speed", "ball_speed"] == 80
    assert result.adjusted_p_values is not None
    assert result.partial_coefficients is not None
    assert "smash_factor" in result.derived_metrics
    assert result.edges


def test_result_is_a_symmetric_matrix_not_a_star(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Every pair is estimated, unlike the outcome-versus-predictors counterpart.

    ``rate_of_closure._launch_monitor_analysis_statistics.correlations``
    estimates one outcome against a list of predictors. This module estimates
    the full upper triangle and mirrors it, which is why the two are not
    interchangeable and why neither package re-exports the other.
    """
    result = compute_correlations(shots(60), metrics=_METRICS)
    assert list(result.coefficients.index) == list(_METRICS)
    assert list(result.coefficients.columns) == list(_METRICS)
    for left in _METRICS:
        assert result.coefficients.loc[left, left] == 1.0
        for right in _METRICS:
            assert result.coefficients.loc[left, right] == pytest.approx(
                result.coefficients.loc[right, left], nan_ok=True
            )


def test_edges_are_screened_and_flag_identity_derived_pairs(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Edges survive both thresholds, and identity pairs are labelled, not dropped."""
    result = compute_correlations(
        shots(80), metrics=_METRICS, edge_threshold=0.3, alpha=0.05
    )
    assert result.edges
    for edge in result.edges:
        assert abs(edge.coefficient) >= 0.3
        assert edge.adjusted_p_value <= 0.05
        expected = "smash_factor" in {edge.source, edge.target}
        assert edge.includes_derived_metric is expected
    strengths = [abs(edge.coefficient) for edge in result.edges]
    assert strengths == sorted(strengths, reverse=True)


def test_raising_the_edge_threshold_only_removes_edges(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """The screen is monotone: a stricter threshold is a subset."""
    frame = shots(80)
    loose = compute_correlations(frame, metrics=_METRICS, edge_threshold=0.1)
    strict = compute_correlations(frame, metrics=_METRICS, edge_threshold=0.9)
    loose_pairs = {(edge.source, edge.target) for edge in loose.edges}
    strict_pairs = {(edge.source, edge.target) for edge in strict.edges}
    assert strict_pairs <= loose_pairs


def test_partial_correlations_are_absent_unless_controls_are_given(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """No controls means no partial matrix — an empty one would imply zero effect."""
    result = compute_correlations(shots(60), metrics=_METRICS)
    assert result.partial_coefficients is None
    controlled = compute_correlations(
        shots(60), metrics=_METRICS, controls=("attack_angle",)
    )
    assert controlled.partial_coefficients is not None
    assert controlled.partial_coefficients.loc["club_speed", "club_speed"] == 1.0


def test_relationships_refuse_one_metric_absent_columns_and_unknown_methods(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """A single metric is not a relationship, and inputs are named, not coerced."""
    frame = shots(20)
    with pytest.raises(ValueError, match=r"At least two metrics are required"):
        compute_correlations(frame, metrics=("club_speed",))
    with pytest.raises(ValueError, match=r"Columns not present"):
        compute_correlations(frame, metrics=("club_speed", "not_a_column"))
    with pytest.raises(ValueError, match=r"Columns not present"):
        compute_correlations(
            frame, metrics=("club_speed", "ball_speed"), controls=("not_a_column",)
        )
    with pytest.raises(ValueError, match=r"pearson, spearman, kendall"):
        compute_correlations(
            frame, metrics=("club_speed", "ball_speed"), method="cosine"
        )


def test_undersampled_pair_yields_nan_and_leaves_the_fdr_denominator(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Today's under-sampling rule, pinned ahead of ruling **D15**.

    The only floor in this port is the hardcoded three complete pairs plus a
    two-distinct-value requirement per side. A pair below it returns ``nan``
    for coefficient and p-value, and because the Benjamini-Hochberg pass keeps
    only finite p-values, such a pair is already outside the correction's
    denominator. D15 (FDR excludes under-sampled predictors before correcting)
    is accepted and lands in a follow-up PR against this module; this
    assertion is the "before" side of that diff.
    """
    frame = shots(20)
    frame["sparse"] = np.nan
    frame.loc[0, "sparse"] = 1.0
    frame.loc[1, "sparse"] = 2.0
    result = compute_correlations(frame, metrics=("club_speed", "ball_speed", "sparse"))
    assert result.pair_counts.loc["club_speed", "sparse"] == 2
    assert np.isnan(result.coefficients.loc["club_speed", "sparse"])
    assert np.isnan(result.p_values.loc["club_speed", "sparse"])
    assert result.adjusted_p_values is not None
    assert np.isnan(result.adjusted_p_values.loc["club_speed", "sparse"])
    assert all("sparse" not in {edge.source, edge.target} for edge in result.edges)


def test_boolean_column_is_silently_projected_to_zero_one(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Today's boolean handling, pinned ahead of ruling **D17**.

    The ``float`` cast inside ``_pair_correlation`` projects a boolean column
    to 0/1 and analyses it as numeric, and nothing in the result records that
    a projection happened. D17 (booleans analysed as 0/1 with explicit
    projection labelling) is accepted and lands in a follow-up PR against this
    module; the projection stays, the silence is what changes. This assertion
    is the "before" side of that diff.
    """
    frame = shots(40)
    frame["is_trackman"] = frame["monitor_vendor"] == "TrackMan"
    boolean = compute_correlations(frame, metrics=("club_speed", "is_trackman"))
    projected = compute_correlations(
        frame.assign(is_trackman=frame["is_trackman"].astype(float)),
        metrics=("club_speed", "is_trackman"),
    )
    assert boolean.coefficients.loc["club_speed", "is_trackman"] == pytest.approx(
        projected.coefficients.loc["club_speed", "is_trackman"]
    )
    assert boolean.pair_counts.loc["club_speed", "is_trackman"] == 40
    assert not hasattr(boolean, "projected_metrics")
    assert not any(
        hasattr(edge, "projection") or hasattr(edge, "projected_from")
        for edge in boolean.edges
    )
