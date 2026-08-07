"""Player-facing launch-monitor analytics contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_data import (
    AnalysisProject,
    campaign_dataset_catalog,
    infer_unit,
    load_analysis_project,
    load_campaign_dataset,
    save_analysis_project,
)
from rate_of_closure.launch_monitor_player_metrics import (
    analyze_dispersion,
    analyze_sessions,
    calculate_strokes_gained_proxy,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_campaign_catalog_exposes_all_manifested_csvs(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    source = tmp_path / "data" / "source.csv"
    source.write_text("shot_id,carry_m\n,[m]\na,1.0\nb,2.0\n", encoding="utf-8")
    pd.DataFrame({"shot_id": ["a", "b"], "carry_m": [1.0, 2.0]}).to_csv(
        tmp_path / "data" / "normalized.csv", index=False
    )
    pd.DataFrame({"shot_id": ["a"], "model": ["one"]}).to_csv(
        tmp_path / "results" / "pred.csv", index=False
    )
    normalized = tmp_path / "data" / "normalized.csv"
    predictions = tmp_path / "results" / "pred.csv"
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    source_hash = digest(source)
    (tmp_path / "campaign.toml").write_text(
        f'[source]\ncsv="data/source.csv"\nexpected_sha256="{source_hash}"\n'
        '[outputs]\nnormalized="data/normalized.csv"\n'
        'predictions="results/pred.csv"\n',
        encoding="utf-8",
    )
    (tmp_path / "results" / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": source_hash,
                "output_sha256": {
                    "normalized": digest(normalized),
                    "predictions": digest(predictions),
                },
            }
        ),
        encoding="utf-8",
    )

    catalog = campaign_dataset_catalog(tmp_path)

    assert [item.dataset_id for item in catalog.datasets] == [
        "source",
        "normalized",
        "predictions",
    ]
    assert [item.row_count for item in catalog.datasets] == [2, 2, 1]
    assert catalog.source_sha256 == source_hash
    source_frame = load_campaign_dataset(catalog.datasets[0])
    assert source_frame["shot_id"].tolist() == ["a", "b"]

    normalized.write_text("shot_id,carry_m\na,9.0\nb,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_campaign_dataset(catalog.datasets[1])


def test_project_round_trip_preserves_source_and_selections(tmp_path: Path) -> None:
    project = AnalysisProject(
        campaign_root="C:/campaign",
        dataset_id="normalized",
        source_sha256="a" * 64,
        selections={"outcome": "carry_m", "predictors": ["ball_speed_mph"]},
        dataset_sha256="b" * 64,
        data_path="C:/campaign/data/normalized.csv",
    )
    path = tmp_path / "analysis.lmproject.json"

    save_analysis_project(path, project)

    assert load_analysis_project(path) == project


def test_units_dispersion_strokes_gained_and_session_trends() -> None:
    frame = pd.DataFrame(
        {
            "player_id": ["p"] * 6,
            "session_id": ["s1"] * 3 + ["s2"] * 3,
            "observed_carry_m": [190, 195, 200, 205, 210, 215],
            "observed_lateral_m": [-9, -3, 0, 2, 5, 11],
            "ball_speed_mph": [140, 141, 142, 145, 146, 147],
        }
    )

    dispersion = analyze_dispersion(frame, "observed_lateral_m", "observed_carry_m")
    assert infer_unit("ball_speed_mph") == "mph"
    assert dispersion.backing_data["lateral_yd"].iloc[0] == pytest.approx(
        -9 * 1.0936133
    )
    assert dispersion.left_count == 2
    assert dispersion.right_count == 3
    assert dispersion.sample_count == 6

    gained = calculate_strokes_gained_proxy(
        frame,
        carry_column="observed_carry_m",
        lateral_column="observed_lateral_m",
        target_distance_yd=240.0,
        start_lie="tee",
        end_lie="fairway",
    )
    assert gained.sample_count == 6
    assert {
        "carry_yd",
        "lateral_yd",
        "remaining_distance_yd",
        "strokes_gained_proxy",
    }.issubset(gained.backing_data.columns)
    assert "Broadie" in gained.method_description

    sessions = analyze_sessions(
        frame,
        metric_column="ball_speed_mph",
        session_column="session_id",
        player_column="player_id",
    )
    assert len(sessions.summary) == 2
    assert sessions.trend_slope_per_session > 0
    assert sessions.summary["metric_unit"].eq("mph").all()


def test_session_trends_are_fitted_independently_by_player() -> None:
    frame = pd.DataFrame(
        {
            "player_id": ["A", "A", "B", "B"],
            "session_id": ["one", "two", "one", "two"],
            "recorded_at": [
                "2026-01-01",
                "2026-01-11",
                "2026-02-01",
                "2026-02-06",
            ],
            "ball_speed_mph": [100.0, 110.0, 200.0, 190.0],
        }
    )

    analysis = analyze_sessions(
        frame,
        metric_column="ball_speed_mph",
        session_column="session_id",
        player_column="player_id",
        time_column="recorded_at",
    )

    slopes = analysis.summary.groupby("player_id")["trend_slope_per_session"].first()
    assert slopes.to_dict() == {"A": pytest.approx(10.0), "B": pytest.approx(-10.0)}
    assert analysis.trend_slope_per_session == pytest.approx(0.0)
    daily_slopes = analysis.summary.groupby("player_id")["trend_slope_per_day"].first()
    assert daily_slopes.to_dict() == {
        "A": pytest.approx(1.0),
        "B": pytest.approx(-2.0),
    }
    assert analysis.trend_slope_per_day == pytest.approx(-0.5)
    assert analysis.summary.groupby("player_id")["session_sequence"].apply(
        list
    ).to_dict() == {"A": [1, 2], "B": [1, 2]}
