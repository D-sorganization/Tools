"""Canonical flexible-analysis contract tests (ADR-0046 G1 step P10).

The first eight cases are UpstreamDrift's
``tests/unit/launch_monitor/test_flexible_analysis.py``, travelling verbatim
with the module they exercise. The remaining cases pin the module's refusals
per this repo's design-by-contract standard. Two of them,
``test_undersampled_predictor_excluded_from_the_fdr_denominator`` and
``test_boolean_predictor_projection_is_labelled_and_math_is_unchanged``,
used to pin the "before" side of owner rulings **D15** and **D17** (ADR-0048,
"Owner Rulings (2026-09-02)"); this module now applies both, so they assert
the "after" contract instead.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from shared.python.launch_monitor.flexible_analysis import (
    CONTRACT_VERSION,
    FlexibleAnalysisRequest,
    analyze_variables,
)
from shared.python.launch_monitor.relationships import compute_correlations

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _shots(count: int = 120) -> pd.DataFrame:
    """UpstreamDrift's frame builder for this file, carried over unchanged."""
    rng = np.random.default_rng(7)
    club_speed = np.linspace(35.0, 52.0, count)
    attack_angle = rng.normal(-0.03, 0.02, count)
    ball_speed = 1.48 * club_speed + 3.2 * attack_angle
    ball_speed += rng.normal(0.0, 0.35, count)
    carry_distance = 3.35 * ball_speed + rng.normal(0.0, 1.5, count)
    return pd.DataFrame(
        {
            "shot_id": [f"shot-{index}" for index in range(count)],
            "session_id": np.where(np.arange(count) < count / 2, "a", "b"),
            "monitor_vendor": np.where(np.arange(count) % 2, "Garmin", "TrackMan"),
            "source_row": np.arange(2, count + 2),
            "club_speed": club_speed,
            "attack_angle": attack_angle,
            "ball_speed": ball_speed,
            "carry_distance": carry_distance,
            "source::custom_numeric": club_speed * 2.0,
        }
    )


# ---------------------------------------------------------------------------
# Ported verbatim from UpstreamDrift's test_flexible_analysis.py
# ---------------------------------------------------------------------------


def test_comprehensive_analysis_reports_uncertainty_diagnostics_and_lineage() -> None:
    result = analyze_variables(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            correlation_method="pearson",
            analysis_mode="comprehensive",
            min_samples=20,
        ),
    )

    assert result.dataset.row_count == 120
    assert result.dataset.complete_row_count == 120
    assert result.dataset.monitor_vendors == ("Garmin", "TrackMan")
    assert len(result.dataset.fingerprint_sha256) == 64
    assert result.regression is not None
    assert result.regression.r_squared > 0.99
    assert result.regression.adjusted_r_squared > 0.99
    assert result.regression.residual_diagnostics.rmse < 0.5
    assert result.regression.coefficients["club_speed"].estimate == pytest.approx(
        1.48, rel=0.02
    )
    assert all(
        item.ci_lower <= item.coefficient <= item.ci_upper
        for item in result.correlations
    )
    adjusted = {item.predictor: item.adjusted_p_value for item in result.correlations}
    assert adjusted["club_speed"] <= 0.05
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())
    payload = result.to_dict()
    assert payload["request"]["outcome"] == "ball_speed"
    assert payload["dataset"]["fingerprint_sha256"] == result.dataset.fingerprint_sha256


def test_pairwise_missing_policy_preserves_per_relationship_counts() -> None:
    frame = _shots(40)
    frame.loc[:4, "attack_angle"] = np.nan
    frame.loc[5:7, "club_speed"] = np.nan

    result = analyze_variables(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            analysis_mode="correlation",
            missing_policy="pairwise",
            min_samples=10,
        ),
    )

    counts = {item.predictor: item.sample_count for item in result.correlations}
    assert counts == {"club_speed": 37, "attack_angle": 35}
    assert result.dataset.complete_row_count == 32


def test_dataset_fingerprint_matches_tools_contract_and_ignores_frame_index() -> None:
    frame = pd.DataFrame(
        {
            "shot_id": ["a", "b", "c"],
            "session_id": ["s", "s", "s"],
            "monitor_vendor": ["TrackMan", "TrackMan", "TrackMan"],
            "x": [1.0, 2.0, 3.0],
            "y": [2.0, 4.0, 6.1],
        },
        index=[10, 20, 30],
    )
    request = FlexibleAnalysisRequest(
        outcome="y",
        predictors=("x",),
        analysis_mode="correlation",
        min_samples=3,
    )

    indexed = analyze_variables(frame, request)
    reset = analyze_variables(frame.reset_index(drop=True), request)

    expected_tools_fingerprint = (
        "6bdb2a22ab06a0fac0b7b0a085f099783759a109d1c38d49eb23f3473c9efff9"
    )
    assert indexed.dataset.fingerprint_sha256 == expected_tools_fingerprint
    assert reset.dataset.fingerprint_sha256 == expected_tools_fingerprint
    assert indexed.to_dict()["contract_version"] == "1.0.0"


def test_grouped_analysis_keeps_monitor_results_separate() -> None:
    result = analyze_variables(
        _shots(),
        FlexibleAnalysisRequest(
            outcome="carry_distance",
            predictors=("ball_speed",),
            group_by="monitor_vendor",
            analysis_mode="comprehensive",
            min_samples=20,
        ),
    )

    assert [group.group_value for group in result.groups] == ["Garmin", "TrackMan"]
    assert all(group.regression is not None for group in result.groups)
    assert all(group.row_count == 60 for group in result.groups)


def test_custom_source_field_cannot_be_pooled_across_monitor_vendors() -> None:
    with pytest.raises(ValueError, match="source fields.*multiple monitors"):
        analyze_variables(
            _shots(),
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("source::custom_numeric",),
                analysis_mode="correlation",
            ),
        )


def test_aggregate_records_never_enter_regression() -> None:
    frame = _shots(30)
    frame["observation_kind"] = "aggregate"
    with pytest.raises(
        ValueError, match="Aggregate observations cannot enter regression"
    ):
        analyze_variables(
            frame,
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed",),
                analysis_mode="regression",
                allow_aggregate=True,
            ),
        )


def test_explicit_aggregate_correlation_is_labeled_descriptive() -> None:
    frame = _shots(30)
    frame["observation_kind"] = "aggregate"
    result = analyze_variables(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="correlation",
            allow_aggregate=True,
        ),
    )

    assert any("ecological" in warning.lower() for warning in result.warnings)


@pytest.mark.parametrize(
    ("analysis_request", "message"),
    [
        (
            FlexibleAnalysisRequest(outcome="ball_speed", predictors=("ball_speed",)),
            "outcome cannot also be a predictor",
        ),
        (
            FlexibleAnalysisRequest(outcome="ball_speed", predictors=("constant",)),
            "Constant variables",
        ),
    ],
)
def test_invalid_analysis_contracts_fail_closed(
    analysis_request: FlexibleAnalysisRequest, message: str
) -> None:
    frame = _shots(30)
    frame["constant"] = 1.0
    with pytest.raises(ValueError, match=message):
        analyze_variables(frame, analysis_request)


# ---------------------------------------------------------------------------
# Owner rulings D15 and D17 applied (ADR-0048, "Owner Rulings (2026-09-02)")
# ---------------------------------------------------------------------------


def test_undersampled_predictor_excluded_from_the_fdr_denominator() -> None:
    """D15 applied: an under-sampled predictor no longer inflates the FDR.

    This is the divergence UpstreamDrift#9372 pinned as **D15**. Before this
    ruling landed, ``_correlations`` read every requested predictor's raw p
    value out of
    :func:`~shared.python.launch_monitor.relationships.compute_correlations`
    and ran Benjamini-Hochberg over the whole pool *before* blanking the
    estimates whose pair count fell below ``min_samples``. A predictor with
    five complete pairs cleared ``relationships``' own three-pair floor,
    contributed a finite raw p to the correction, and only afterwards reported
    ``nan`` — so the three fully sampled predictors were corrected against k=4
    instead of k=3 and their adjusted p values came back inflated by exactly
    4/3. The ruling excludes under-sampled predictors from the denominator
    *before* correcting, so requesting the under-sampled predictor alongside
    the sampled ones must no longer change the sampled ones' adjusted p
    values at all — asking for a fourth, unusable predictor is now a no-op
    for the three that matter, which is the property this test now proves
    instead of the inflation it used to pin.
    """
    frame = _shots(60)
    frame["sparse_metric"] = np.nan
    frame.loc[frame.index[:5], "sparse_metric"] = np.arange(5.0)
    settings = {
        "outcome": "ball_speed",
        "analysis_mode": "correlation",
        "min_samples": 10,
    }
    sampled = ("club_speed", "attack_angle", "carry_distance")

    with_sparse = analyze_variables(
        frame,
        FlexibleAnalysisRequest(predictors=(*sampled, "sparse_metric"), **settings),  # type: ignore[arg-type]
    )
    without_sparse = analyze_variables(
        frame,
        FlexibleAnalysisRequest(predictors=sampled, **settings),  # type: ignore[arg-type]
    )

    with_by_predictor = {item.predictor: item for item in with_sparse.correlations}
    without_by_predictor = {
        item.predictor: item for item in without_sparse.correlations
    }

    # The under-sampled predictor still reports nothing at all ...
    sparse = with_by_predictor["sparse_metric"]
    assert sparse.sample_count == 5
    assert np.isnan(sparse.coefficient)
    assert np.isnan(sparse.p_value)
    assert np.isnan(sparse.adjusted_p_value)
    assert not sparse.is_boolean_projected

    # ... and, after this ruling, it no longer moves anyone else's denominator:
    # the fourth predictor's presence is now invisible to the other three.
    for predictor in sampled:
        included = with_by_predictor[predictor]
        excluded = without_by_predictor[predictor]
        assert included.sample_count == excluded.sample_count == 60
        # Identical raw p values on both sides: this was already true, and
        # still is - only the correction denominator was ever in play.
        assert included.p_value - excluded.p_value == 0.0
        # The math this ruling is about: the adjusted p value is now
        # unaffected by whether the under-sampled predictor was requested.
        assert included.adjusted_p_value == excluded.adjusted_p_value

    # The two surviving pins are unchanged by this ruling - they were always
    # the k=3, exclude-the-sparse-one values; only the "with_sparse" side
    # used to diverge from them, by exactly 4/3.
    assert without_by_predictor["club_speed"].adjusted_p_value == pytest.approx(
        6.351788337665487e-82, rel=1e-12
    )
    assert with_by_predictor["club_speed"].adjusted_p_value == pytest.approx(
        6.351788337665487e-82, rel=1e-12
    )
    assert without_by_predictor["carry_distance"].adjusted_p_value == pytest.approx(
        4.809382390570119e-75, rel=1e-12
    )
    assert with_by_predictor["carry_distance"].adjusted_p_value == pytest.approx(
        4.809382390570119e-75, rel=1e-12
    )

    # The blanked estimate still serialises as JSON null.
    payload = with_sparse.to_dict()
    serialised = {item["predictor"]: item for item in payload["correlations"]}
    assert serialised["sparse_metric"]["adjusted_p_value"] is None


def test_boolean_predictor_projection_is_labelled_and_math_is_unchanged() -> None:
    """D17 applied: the projection label now survives up from ``relationships``.

    ``analyze_variables`` still runs the selected columns through
    ``pd.to_numeric``, which still projects ``True``/``False`` to 1.0/0.0. The
    column still passes the two-distinct-value constancy screen, is still
    correlated as though it were a native numeric metric, and still counts
    every row toward the sample count — the capability D17 preserves, and the
    coefficient below is bit-identical to what this test pinned before the
    ruling landed.

    What changed is the silence. Tools#4901 applied D17 to
    :mod:`~shared.python.launch_monitor.relationships`, so
    ``compute_correlations`` reports the projection in
    ``CorrelationResult.boolean_projected``; ``_correlations`` now reads that
    label off the result it already holds and carries it onto each
    :class:`CorrelationEstimate` as ``is_boolean_projected`` instead of
    dropping it. Deliberately unchanged by this ruling: the unit map (``units``
    resolution is D23's territory, applied elsewhere) and ``warnings`` (a
    boolean projection is not treated as an error condition, matching the
    ruling's "capability preserved" framing).
    """
    frame = _shots(60)
    frame["flagged"] = np.arange(60) % 2 == 0
    assert frame["flagged"].dtype == np.dtype(bool)

    result = analyze_variables(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "flagged"),
            analysis_mode="correlation",
            min_samples=10,
        ),
    )

    flagged = next(item for item in result.correlations if item.predictor == "flagged")
    club_speed = next(
        item for item in result.correlations if item.predictor == "club_speed"
    )
    assert flagged.sample_count == 60
    assert flagged.coefficient == pytest.approx(-0.029183486713892384, rel=1e-12)
    assert result.dataset.complete_row_count == 60

    # The layer below already knows, and computes the identical coefficient.
    underlying = compute_correlations(
        frame, metrics=("ball_speed", "club_speed", "flagged")
    )
    assert underlying.boolean_projected == ("flagged",)
    assert underlying.coefficients.loc["ball_speed", "flagged"] == flagged.coefficient

    # The label is now explicit: the boolean-projected predictor is marked,
    # a native-numeric predictor never is, and the unit/warning surfaces this
    # ruling does not touch are unchanged.
    assert flagged.is_boolean_projected is True
    assert club_speed.is_boolean_projected is False
    assert result.units["flagged"] == "source"
    assert result.warnings == ()
    assert "is_boolean_projected" in type(flagged).__dataclass_fields__
    assert not any(
        "boolean" in field.lower() for field in type(result).__dataclass_fields__
    )
    payload = result.to_dict()
    serialised = {item["predictor"]: item for item in payload["correlations"]}
    assert set(serialised["flagged"]) == set(serialised["club_speed"])
    assert serialised["flagged"]["is_boolean_projected"] is True
    assert serialised["club_speed"]["is_boolean_projected"] is False


# ---------------------------------------------------------------------------
# Design-by-contract refusals (CLAUDE.md: every public function validates input)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"outcome": "   "}, "outcome must be non-empty"),
        ({"predictors": ()}, "At least one predictor is required"),
        (
            {"predictors": ("club_speed", "club_speed")},
            "predictors must be unique",
        ),
        ({"analysis_mode": "bogus"}, "Unknown analysis_mode"),
        ({"correlation_method": "bogus"}, "Unknown correlation_method"),
        ({"missing_policy": "bogus"}, "Unknown missing_policy"),
        ({"confidence_level": 0.5}, "confidence_level must be between 0.5 and 1"),
        ({"confidence_level": 1.0}, "confidence_level must be between 0.5 and 1"),
        ({"min_samples": 2}, "min_samples must be at least 3"),
    ],
)
def test_request_validates_every_field_at_construction(
    overrides: dict[str, object], message: str
) -> None:
    """The request fails closed on construction, never at analysis time.

    This is the half of divergence **D16** that lives on this side: the
    ``rate_of_closure`` twin validates none of these three enums and silently
    degrades an unknown ``correlation_method`` into Kendall while still
    reporting the caller's spelling.
    """
    settings: dict[str, object] = {
        "outcome": "ball_speed",
        "predictors": ("club_speed",),
    }
    settings.update(overrides)
    with pytest.raises(ValueError, match=message):
        FlexibleAnalysisRequest(**settings)  # type: ignore[arg-type]


def test_absent_columns_are_named_in_the_refusal() -> None:
    with pytest.raises(ValueError, match=r"Columns not present.*missing_metric"):
        analyze_variables(
            _shots(30),
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("missing_metric",),
                analysis_mode="correlation",
            ),
        )


def test_absent_group_by_column_is_a_refusal_not_a_silent_ungrouped_result() -> None:
    with pytest.raises(ValueError, match=r"Columns not present.*no_such_group"):
        analyze_variables(
            _shots(30),
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed",),
                analysis_mode="correlation",
                group_by="no_such_group",
            ),
        )


def test_fail_missing_policy_refuses_any_gap() -> None:
    frame = _shots(30)
    frame.loc[frame.index[0], "attack_angle"] = np.nan
    with pytest.raises(ValueError, match="missing or non-numeric"):
        analyze_variables(
            frame,
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed", "attack_angle"),
                analysis_mode="correlation",
                missing_policy="fail",
            ),
        )


def test_non_numeric_text_is_a_gap_under_fail_policy_too() -> None:
    """A text column is coerced to ``nan``, so ``fail`` refuses it as a gap."""
    frame = _shots(30)
    frame["texty"] = [f"{value:.4f}" for value in frame["club_speed"]]
    frame.loc[frame.index[0], "texty"] = "0x10"
    with pytest.raises(ValueError, match="missing or non-numeric"):
        analyze_variables(
            frame,
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("texty",),
                analysis_mode="correlation",
                missing_policy="fail",
            ),
        )


def test_regression_refuses_too_few_complete_observations() -> None:
    """The floor is ``max(min_samples, parameter_count + 2)``, not ``min_samples``.

    Four rows against a three-parameter design (intercept plus two predictors)
    is below the five the fit needs, even though ``min_samples`` is only 3.
    """
    with pytest.raises(ValueError, match="Too few complete observations"):
        analyze_variables(
            _shots(4),
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed", "attack_angle"),
                analysis_mode="regression",
                min_samples=3,
            ),
        )


def test_regression_refuses_a_rank_deficient_design() -> None:
    frame = _shots(40)
    frame["club_speed_copy"] = frame["club_speed"]
    with pytest.raises(ValueError, match="rank deficient"):
        analyze_variables(
            frame,
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed", "club_speed_copy"),
                analysis_mode="regression",
                min_samples=10,
            ),
        )


def test_aggregate_records_require_explicit_opt_in_even_for_correlation() -> None:
    frame = _shots(30)
    frame["observation_kind"] = "aggregate"
    with pytest.raises(ValueError, match="require allow_aggregate=True"):
        analyze_variables(
            frame,
            FlexibleAnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed",),
                analysis_mode="correlation",
            ),
        )


def test_a_failing_group_is_reported_not_raised() -> None:
    """One unusable group must not discard the groups that did analyse."""
    frame = _shots(60)
    frame["cohort"] = np.where(np.arange(60) < 6, "tiny", "bulk")
    frame.loc[frame.index[:6], "club_speed"] = 40.0

    result = analyze_variables(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="correlation",
            group_by="cohort",
            min_samples=10,
        ),
    )

    by_value = {group.group_value: group for group in result.groups}
    assert set(by_value) == {"bulk", "tiny"}
    assert by_value["bulk"].correlations
    assert by_value["tiny"].correlations == ()
    assert by_value["tiny"].warnings
    assert "Constant variables" in by_value["tiny"].warnings[0]


def test_non_finite_estimates_never_reach_the_wire_as_nan() -> None:
    """``to_dict`` refuses to emit non-JSON ``NaN`` tokens."""
    frame = _shots(20)
    frame.loc[frame.index[4:], "attack_angle"] = np.nan
    result = analyze_variables(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            analysis_mode="correlation",
            min_samples=10,
        ),
    )

    payload = result.to_dict()
    assert payload["contract_version"] == CONTRACT_VERSION == "1.0.0"
    serialised = {item["predictor"]: item for item in payload["correlations"]}
    assert serialised["attack_angle"]["coefficient"] is None
    assert serialised["attack_angle"]["p_value"] is None
    # ``allow_nan=False`` is the strict encoder a wire boundary uses: it raises
    # rather than emitting the non-standard ``NaN`` token.
    assert json.dumps(payload, allow_nan=False)


def test_units_come_from_the_canonical_registry_or_are_labelled_source() -> None:
    known = analyze_variables(
        _shots(40),
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="correlation",
            min_samples=10,
        ),
    )
    assert known.units == {"ball_speed": "m/s", "club_speed": "m/s"}

    frame = _shots(40)
    frame["monitor_vendor"] = "TrackMan"
    unknown = analyze_variables(
        frame,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("source::custom_numeric",),
            analysis_mode="correlation",
            min_samples=10,
        ),
    )
    assert unknown.units["source::custom_numeric"] == "source"
