"""Within-player covariation and cross-player meta-analysis contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rate_of_closure.player_covariation import (
    CovariationRequest,
    PairScanRequest,
    analyze_player_covariation,
    scan_covariation_pairs,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _confounded_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player": ["A"] * 5 + ["B"] * 5,
            "face_angle_deg": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
            "club_path_deg": [4, 3, 2, 1, 0, 14, 13, 12, 11, 10],
        }
    )


def test_separates_pooled_within_and_between_association() -> None:
    analysis = analyze_player_covariation(
        _confounded_frame(),
        CovariationRequest("face_angle_deg", "club_path_deg", "player"),
    )

    assert analysis.pooled.pearson_r == pytest.approx(0.8518518519)
    assert analysis.within_player.pearson_r == pytest.approx(-1.0)
    assert analysis.within_player.ci_lower is None
    assert analysis.within_player.ci_upper is None
    assert analysis.between_player.pearson_r == pytest.approx(1.0)
    assert analysis.within_player.intercept == pytest.approx(0.0)
    assert analysis.meta_analysis.contributor_count == 2
    assert analysis.meta_analysis.fixed_effect_r == pytest.approx(-1.0, abs=1e-5)
    assert analysis.units == {"x": "deg", "y": "deg"}
    assert "does not imply causation" in analysis.method_description
    assert {"centered_x", "centered_y"}.issubset(analysis.backing_data.columns)
    assert any("aggregation reversal" in warning for warning in analysis.warnings)


def test_player_rows_include_fisher_intervals_regression_and_weights() -> None:
    analysis = analyze_player_covariation(
        _confounded_frame(),
        CovariationRequest("face_angle_deg", "club_path_deg", "player"),
    )

    assert list(analysis.per_player["player_id"]) == ["A", "B"]
    assert analysis.per_player["sample_count"].tolist() == [5, 5]
    assert analysis.per_player["spearman_r"].tolist() == pytest.approx([-1, -1])
    assert analysis.per_player["slope"].tolist() == pytest.approx([-1, -1])
    assert analysis.per_player["intercept"].tolist() == pytest.approx([4, 24])
    assert analysis.per_player["r_squared"].tolist() == pytest.approx([1, 1])
    assert analysis.per_player["ci_lower"].notna().all()
    assert analysis.per_player["fixed_weight"].sum() == pytest.approx(1.0)
    assert analysis.per_player["random_weight"].sum() == pytest.approx(1.0)


def test_missing_small_and_constant_groups_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "player": ["good"] * 4 + ["small"] * 3 + ["constant"] * 4,
            "x_mph": [1, 2, 3, 4, 1, 2, np.nan, 5, 5, 5, 5],
            "y_deg": [2, 4, 6, 8, 2, 4, 6, 1, 2, 3, 4],
        }
    )

    analysis = analyze_player_covariation(
        frame, CovariationRequest("x_mph", "y_deg", "player", min_samples=4)
    )

    statuses = dict(
        zip(
            analysis.per_player.player_id,
            analysis.per_player.status,
            strict=True,
        )
    )
    assert statuses == {
        "constant": "constant_x",
        "good": "ok",
        "small": "insufficient_samples",
    }
    invalid = analysis.per_player.query("status != 'ok'")
    assert invalid["pearson_r"].isna().all()
    assert analysis.meta_analysis.contributor_count == 1
    assert any("excluded" in warning for warning in analysis.warnings)
    assert analysis.meta_analysis.fixed_effect_r is None
    assert any("two eligible players" in warning for warning in analysis.warnings)


def test_blank_player_identity_is_not_created_as_a_group() -> None:
    frame = pd.DataFrame(
        {
            "player": ["A"] * 4 + ["", "  ", None, np.nan],
            "x": range(8),
            "y": range(8),
        }
    )

    analysis = analyze_player_covariation(frame, CovariationRequest("x", "y", "player"))

    assert analysis.per_player["player_id"].tolist() == ["A"]
    assert analysis.backing_data["player_id"].unique().tolist() == ["A"]
    assert any("4 rows" in warning for warning in analysis.warnings)
    assert "Spearman is descriptive" in analysis.method_description


def test_der_simonian_laird_reports_heterogeneity() -> None:
    rows: list[dict[str, object]] = []
    x_values = np.arange(8, dtype=float)
    for player, y_values in {
        "positive": x_values,
        "negative": -x_values,
        "moderate": np.array([0, 2, 1, 4, 3, 6, 5, 7], dtype=float),
    }.items():
        rows.extend(
            {"player": player, "x": x, "y": y}
            for x, y in zip(x_values, y_values, strict=True)
        )

    analysis = analyze_player_covariation(
        pd.DataFrame(rows), CovariationRequest("x", "y", "player")
    )

    meta = analysis.meta_analysis
    assert meta.contributor_count == 3
    assert meta.total_sample_count == 24
    assert meta.q_statistic > 0
    assert meta.tau_squared > 0
    assert meta.i_squared_pct > 0
    assert meta.random_ci_lower < meta.random_effect_r < meta.random_ci_upper


def test_pair_scan_is_deterministic_and_warns_about_multiplicity() -> None:
    frame = _confounded_frame().assign(
        ball_speed_mph=[100, 102, 104, 106, 108, 120, 122, 124, 126, 128]
    )
    request = PairScanRequest(
        player_column="player",
        numeric_columns=("club_path_deg", "ball_speed_mph", "face_angle_deg"),
    )

    result = scan_covariation_pairs(frame, request)

    assert len(result.ranking) == 3
    assert result.ranking.iloc[0]["x_column"] == "ball_speed_mph"
    assert result.ranking.iloc[0]["y_column"] == "club_path_deg"
    assert result.ranking.iloc[0]["random_effect_r"] == pytest.approx(-1.0, abs=1e-5)
    assert result.ranking.iloc[0]["direction_consistency"] == pytest.approx(1.0)
    assert any("exploratory" in warning.lower() for warning in result.warnings)
    assert any("3 pairs" in warning for warning in result.warnings)


def test_default_pair_scan_ignores_non_numeric_columns() -> None:
    frame = _confounded_frame().assign(club=["7i"] * 10)

    result = scan_covariation_pairs(frame, PairScanRequest(player_column="player"))

    assert len(result.ranking) == 1
    assert result.ranking.iloc[0]["x_column"] == "club_path_deg"
    assert result.ranking.iloc[0]["y_column"] == "face_angle_deg"


@pytest.mark.parametrize(
    ("analysis_request", "message"),
    [
        (CovariationRequest("x", "x", "player"), "must differ"),
        (CovariationRequest("x", "y", "missing"), "missing required"),
        (CovariationRequest("x", "y", "player", confidence_level=1.0), "confidence"),
    ],
)
def test_request_contract_validation(
    analysis_request: CovariationRequest, message: str
) -> None:
    frame = pd.DataFrame({"player": ["A"] * 4, "x": range(4), "y": range(4)})
    with pytest.raises(ValueError, match=message):
        analyze_player_covariation(frame, analysis_request)
