"""Inferential player/session longitudinal analysis contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_longitudinal import (
    LongitudinalRequest,
    analyze_longitudinal_performance,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for player, means in (("p1", [100, 103, 104, 108]), ("p2", [90, 91, 94, 94])):
        for order, mean in enumerate(means, start=1):
            for offset in (-0.5, 0.5):
                rows.append(
                    {
                        "player": player,
                        "session": f"{player}-{order}",
                        "order": order,
                        "ball_speed": mean + offset,
                    }
                )
    return pd.DataFrame(rows)


def _request() -> LongitudinalRequest:
    return LongitudinalRequest(
        "ball_speed", "session", "order", "player", True, True, True, 0.95, 3
    )


def test_longitudinal_reports_session_uncertainty_player_slopes_and_population() -> (
    None
):
    result = analyze_longitudinal_performance(_frame(), _request())

    assert len(result.session_points) == 8
    assert result.session_points[0].standard_error == pytest.approx(0.5)
    assert len(result.players) == 2
    assert all(
        player.status == "ok" and player.slope_per_session > 0
        for player in result.players
    )
    assert result.population.contributor_count == 2
    assert result.population.random_effect_slope is not None
    assert result.population.random_effect_slope > 0
    assert result.population.improvement_probability is not None
    assert result.population.improvement_probability > 0.5
    assert "does not establish causality" in " ".join(result.warnings)


def test_longitudinal_fails_closed_on_identity_and_duplicate_session_order() -> None:
    request = _request()
    with pytest.raises(ValueError, match="attested"):
        analyze_longitudinal_performance(
            _frame(),
            LongitudinalRequest(
                request.metric_column,
                request.session_column,
                request.session_order_column,
                request.player_column,
                False,
                True,
                True,
                0.95,
                3,
            ),
        )
    duplicate = _frame()
    duplicate.loc[duplicate["session"] == "p1-2", "order"] = 1
    with pytest.raises(ValueError, match="unique order"):
        analyze_longitudinal_performance(duplicate, request)


def test_longitudinal_retains_small_players_but_excludes_them_from_synthesis() -> None:
    frame = _frame()
    frame = frame[~((frame["player"] == "p2") & (frame["order"] > 2))]
    result = analyze_longitudinal_performance(frame, _request())

    statuses = {player.player_id: player.status for player in result.players}
    assert statuses == {"p1": "ok", "p2": "insufficient_sessions"}
    assert result.population.contributor_count == 1
    assert result.population.random_effect_slope is None


def test_pyqt_longitudinal_widget_renders_units_and_export_payload(qtbot) -> None:  # type: ignore[no-untyped-def]
    from rate_of_closure.ui.pyqt6.launch_monitor_longitudinal_widget import (
        LaunchMonitorLongitudinalWidget,
    )

    widget = LaunchMonitorLongitudinalWidget()
    qtbot.addWidget(widget)
    widget.set_dataset(_frame())
    widget.player_combo.setCurrentText("player")
    widget.session_combo.setCurrentText("session")
    widget.order_combo.setCurrentText("order")
    widget.metric_combo.setCurrentText("ball_speed")
    widget.player_attest.setChecked(True)
    widget.session_attest.setChecked(True)
    result = widget.calculate()

    assert result.population.contributor_count == 2
    assert widget.table.rowCount() == 2
    assert "unknown" in widget.axes.get_ylabel()
    assert widget.document()["session_points"]
