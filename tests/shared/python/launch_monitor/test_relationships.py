"""Canonical relationship-analysis tests (ADR-0046 G1 step P7).

The first case is UpstreamDrift's
``tests/unit/launch_monitor/test_analysis.py::test_correlations_include_counts_significance_and_derived_warning``,
travelling verbatim with the module it exercises. The remaining cases pin the
module's refusals. Owner ruling **D17** — booleans analysed as 0/1 with the
projection explicitly labelled — is applied (UpstreamDrift PR #9392,
``docs/adr/0048-launch-monitor-port-plan.md`` "Owner Rulings (2026-09-02)");
``test_boolean_column_projection_is_labelled_and_math_is_unchanged`` below
asserts the "after" contract in place of the old silent-projection pin. Owner
ruling **D15** (FDR excludes under-sampled predictors before correcting) does
**not** reach this module —
``test_undersampled_pair_yields_nan_and_leaves_the_fdr_denominator`` below is
not a "before" pin; it asserts final, ruling-compliant behaviour that was
already correct by construction (see the module docstring for why). D15's
actual application is in :mod:`~shared.python.launch_monitor.flexible_analysis`.
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
    """This module's under-sampling rule already satisfies ruling **D15**.

    The only floor in this port is the hardcoded three complete pairs plus a
    two-distinct-value requirement per side. A pair below it returns ``nan``
    for coefficient and p-value, and because the Benjamini-Hochberg pass keeps
    only finite p-values, such a pair is already outside the correction's
    denominator. D15 (FDR excludes under-sampled predictors before correcting)
    reads, on a first pass over ADR-0048, as though it reaches this module —
    it does not: there is no separate, larger ``min_samples`` tier here for
    the ruling's defect (a predictor that clears the floor, pollutes the
    correction, and is only afterwards blanked) to exist in. This is
    permanent, ruling-compliant behaviour, not a "before" pin; D15's actual
    fix lands in
    :mod:`~shared.python.launch_monitor.flexible_analysis`, which has exactly
    that second, configurable tier.
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


def test_boolean_column_projection_is_labelled_and_math_is_unchanged(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Ruling **D17** applied: booleans are still analysed as 0/1, but labelled.

    UpstreamDrift PR #9392 / ``docs/adr/0048-launch-monitor-port-plan.md``
    "Owner Rulings (2026-09-02)", D17: the boolean-analysis capability is
    preserved (a boolean column is still projected to 0/1 and analysed), but
    the projection must be explicit in the result — a column analysed via
    boolean projection is labelled as such and can never read as native
    numeric. This replaces the old silent-projection pin: the projection
    itself, and therefore every coefficient/p-value it feeds, is bit-for-bit
    the same as before this change (labelling changes metadata, never math);
    only the presence of an explicit label is new.
    """
    frame = shots(40)
    frame["is_trackman"] = frame["monitor_vendor"] == "TrackMan"
    boolean = compute_correlations(frame, metrics=("club_speed", "is_trackman"))
    projected = compute_correlations(
        frame.assign(is_trackman=frame["is_trackman"].astype(float)),
        metrics=("club_speed", "is_trackman"),
    )

    # Math is unchanged: the boolean-column result is identical to analysing
    # the same values already cast to float, and matches the pinned value
    # this module has always produced for this fixture.
    r = boolean.coefficients.loc["club_speed", "is_trackman"]
    assert r == pytest.approx(projected.coefficients.loc["club_speed", "is_trackman"])
    assert r == pytest.approx(-0.04331480818242096)
    assert boolean.pair_counts.loc["club_speed", "is_trackman"] == 40

    # The projection is now explicit: the boolean metric is named on the
    # result, and a native-numeric column never is.
    assert boolean.boolean_projected == ("is_trackman",)
    assert "club_speed" not in boolean.boolean_projected
    # Casting to float ahead of time means there is no longer a boolean
    # column to project — nothing is labelled.
    assert projected.boolean_projected == ()

    # A native numeric-only analysis carries no boolean label at all.
    numeric_only = compute_correlations(frame, metrics=("club_speed", "ball_speed"))
    assert numeric_only.boolean_projected == ()
    assert not any(edge.includes_boolean_projection for edge in numeric_only.edges)

    # Any edge touching the boolean-projected metric is labelled; edges
    # between two native-numeric metrics never are. r is small enough here
    # that the default edge_threshold screens the pair out, so force edges
    # through with a permissive threshold to exercise the label directly.
    labelled = compute_correlations(
        frame,
        metrics=("club_speed", "ball_speed", "is_trackman"),
        edge_threshold=0.0,
        alpha=1.0,
    )
    assert labelled.edges
    for edge in labelled.edges:
        touches_boolean = "is_trackman" in {edge.source, edge.target}
        assert edge.includes_boolean_projection is touches_boolean
    assert any(edge.includes_boolean_projection for edge in labelled.edges)
