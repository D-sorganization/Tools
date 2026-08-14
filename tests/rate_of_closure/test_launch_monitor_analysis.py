"""Parity and safety contracts for desktop launch-monitor analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rate_of_closure.launch_monitor_analysis import (
    AnalysisRequest,
    analyze_launch_monitor_data,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _frame() -> pd.DataFrame:
    count = 80
    index = np.arange(count)
    club_speed = 35.0 + index * 0.2
    attack_angle = -3.0 + (index % 8) * 0.5
    return pd.DataFrame(
        {
            "shot_id": [f"shot-{item}" for item in index],
            "session_id": np.where(index < 40, "a", "b"),
            "monitor_vendor": np.where(index % 2, "FlightScope", "TrackMan"),
            "club_speed": club_speed,
            "attack_angle": attack_angle,
            "ball_speed": 1.48 * club_speed + 0.04 * attack_angle,
        }
    )


def test_analysis_reports_uncertainty_diagnostics_groups_and_lineage() -> None:
    result = analyze_launch_monitor_data(
        _frame(),
        AnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            group_by="monitor_vendor",
        ),
    )

    assert result.contract_version == "1.0.0"
    assert result.dataset.row_count == 80
    assert result.dataset.monitor_vendors == ("FlightScope", "TrackMan")
    assert len(result.dataset.fingerprint_sha256) == 64
    assert result.regression is not None
    assert result.regression.r_squared > 0.999
    assert result.regression.coefficients["club_speed"].estimate == pytest.approx(1.48)
    assert [group.group_value for group in result.groups] == ["FlightScope", "TrackMan"]
    assert result.to_wire()["contractVersion"] == "1.0.0"


def test_pairwise_missingness_and_fail_closed_boundaries() -> None:
    frame = _frame()
    frame.loc[0, "attack_angle"] = np.nan
    result = analyze_launch_monitor_data(
        frame,
        AnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed", "attack_angle"),
            analysis_mode="correlation",
        ),
    )
    assert {item.predictor: item.sample_count for item in result.correlations} == {
        "club_speed": 80,
        "attack_angle": 79,
    }

    aggregate = _frame().assign(observation_kind="aggregate")
    with pytest.raises(
        ValueError, match="Aggregate observations cannot enter regression"
    ):
        analyze_launch_monitor_data(
            aggregate,
            AnalysisRequest(
                outcome="ball_speed",
                predictors=("club_speed",),
                analysis_mode="regression",
                allow_aggregate=True,
            ),
        )
