"""Phase-aware playback contracts for imported ground results."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from rate_of_closure.simulation.ground_playback import (
    GroundPlaybackTimeline,
    load_ground_result_json,
)
from rate_of_closure.simulation.ground_playback_comparison import (
    GroundPlaybackComparison,
    ground_comparison_csv,
    ground_comparison_json,
)
from rate_of_closure.simulation.ground_playback_workspace import (
    GROUND_PLAYBACK_WORKSPACE_SCHEMA,
    GroundPlaybackState,
    GroundPlaybackViewState,
    GroundPlaybackWorkspace,
    ground_event_csv,
    ground_result_json,
    ground_trajectory_csv,
    ground_workspace_from_json,
    ground_workspace_to_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "ground_reference_pipeline_golden_v1.json"
)


def _result_payload() -> dict[str, object]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload["result"]


def _comparison_payload(*, time_offset_s: float = 0.2) -> dict[str, object]:
    payload = _result_payload()
    payload["request_id"] = "comparison-run"
    payload["provenance"]["input_sha256"] = "b" * 64  # type: ignore[index]
    for point in payload["trajectory"]:  # type: ignore[union-attr]
        point["time_s"] += time_offset_s
    for event in payload["events"]:  # type: ignore[union-attr]
        event["time_s"] += time_offset_s
    payload["termination"]["time_s"] += time_offset_s  # type: ignore[index]
    return payload


def test_strict_import_accepts_only_result_and_builds_absolute_timeline() -> None:
    result = load_ground_result_json(json.dumps(_result_payload()))
    timeline = GroundPlaybackTimeline(result)

    assert timeline.start_time_s == pytest.approx(1.005)
    assert timeline.end_time_s == pytest.approx(1.50466094435)
    assert timeline.duration_s == pytest.approx(0.49966094435)
    assert timeline.end_label == "Rest"
    assert timeline.phase_time("skid") == pytest.approx(1.00907886485)
    assert timeline.phase_time("roll") == pytest.approx(1.14047658257)
    assert timeline.phase_time("rest") == pytest.approx(1.50466094435)

    with pytest.raises((TypeError, ValueError)):
        load_ground_result_json(FIXTURE.read_text(encoding="utf-8"))


def test_phase_transition_holds_lower_sample_until_exact_boundary() -> None:
    timeline = GroundPlaybackTimeline(
        load_ground_result_json(json.dumps(_result_payload()))
    )
    before_roll = timeline.frame_at(1.13)
    at_roll = timeline.frame_at(1.14047658257)

    assert before_roll.phase == "skid"
    assert before_roll.position_m == pytest.approx((0.08819335004, 0.02135, 0.0))
    assert before_roll.interpolation_fraction == 0.0
    assert at_roll.phase == "roll"
    assert at_roll.position_m == pytest.approx((0.11476801928, 0.02135, 0.0))


def test_same_phase_interpolates_and_steps_use_exact_samples() -> None:
    timeline = GroundPlaybackTimeline(
        load_ground_result_json(json.dumps(_result_payload()))
    )
    midpoint = timeline.frame_at((1.205 + 1.305) / 2.0)

    assert midpoint.phase == "roll"
    assert midpoint.position_m[0] == pytest.approx(
        (0.15677340009 + 0.20574015016) / 2.0
    )
    assert midpoint.interpolation_fraction == pytest.approx(0.5)
    assert timeline.step_time(1.205, 1) == pytest.approx(1.305)
    assert timeline.step_time(1.205, -1) == pytest.approx(1.14047658257)
    assert timeline.step_time(timeline.end_time_s, 1) == timeline.end_time_s


def test_markers_and_partial_endpoint_language_are_honest() -> None:
    payload = _result_payload()
    payload["status"] = "partial"
    payload["trajectory"] = payload["trajectory"][:-1]  # type: ignore[index]
    payload["events"] = payload["events"][:-1]  # type: ignore[index]
    final = payload["trajectory"][-1]  # type: ignore[index]
    summary = payload["summary"]  # type: ignore[assignment]
    summary["final_downrange_m"] = final["position_m"][0]  # type: ignore[index]
    summary["final_offline_m"] = final["position_m"][2]  # type: ignore[index]
    summary["total_distance_m"] = final["position_m"][0]  # type: ignore[index]
    payload["termination"] = {
        "completed": False,
        "reason": "time_limit",
        "time_s": final["time_s"],  # type: ignore[index]
    }
    timeline = GroundPlaybackTimeline(load_ground_result_json(json.dumps(payload)))

    assert timeline.carry_position_m == pytest.approx((0.0, 0.02135, 0.0))
    assert timeline.endpoint_position_m == pytest.approx((0.23509360023, 0.02135, 0.0))
    assert timeline.end_label == "Observed end"
    assert not timeline.is_complete


def test_import_is_bounded_and_failed_results_cannot_be_played() -> None:
    encoded = json.dumps(_result_payload())
    with pytest.raises(ValueError, match="size limit"):
        load_ground_result_json(encoded, max_bytes=10)

    payload = _result_payload()
    payload.update(
        status="failed",
        trajectory=[],
        events=[],
        summary=None,
        termination={
            "completed": False,
            "reason": "numerical_failure",
            "time_s": 0.0,
        },
    )
    result = load_ground_result_json(json.dumps(payload))
    with pytest.raises(ValueError, match="complete or partial"):
        GroundPlaybackTimeline(result)


def _workspace() -> GroundPlaybackWorkspace:
    result = load_ground_result_json(json.dumps(_result_payload()))
    return GroundPlaybackWorkspace(
        result=result,
        playback=GroundPlaybackState(time_s=1.205, speed=2.0, loop=True),
        view=GroundPlaybackViewState(yaw_deg=-37.5, pitch_deg=18.0, zoom=1.75),
    )


def test_workspace_round_trip_is_strict_versioned_and_deterministic() -> None:
    workspace = _workspace()
    encoded = ground_workspace_to_json(workspace)
    restored = ground_workspace_from_json(encoded)

    assert encoded == ground_workspace_to_json(restored)
    assert restored == workspace
    assert json.loads(encoded)["schema_version"] == GROUND_PLAYBACK_WORKSPACE_SCHEMA
    assert json.loads(encoded)["result"] == _result_payload()
    assert "playing" not in json.loads(encoded)["playback"]


def test_workspace_rejects_duplicate_unknown_and_invalid_state() -> None:
    encoded = ground_workspace_to_json(_workspace())
    duplicate = encoded.replace(
        '"schema_version":"rate-of-closure-ground-playback-workspace/v1"',
        '"schema_version":"rate-of-closure-ground-playback-workspace/v1",'
        '"schema_version":"duplicate"',
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        ground_workspace_from_json(duplicate)

    payload = json.loads(encoded)
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match"):
        ground_workspace_from_json(json.dumps(payload))

    payload = json.loads(encoded)
    payload["playback"]["time_s"] = 99.0
    with pytest.raises(ValueError, match="within the result timeline"):
        ground_workspace_from_json(json.dumps(payload))

    with pytest.raises(ValueError, match="supported playback speed"):
        GroundPlaybackState(time_s=1.205, speed=3.0, loop=False)
    with pytest.raises(ValueError, match="zoom"):
        GroundPlaybackViewState(yaw_deg=0.0, pitch_deg=0.0, zoom=0.1)
    with pytest.raises(ValueError, match="size limit"):
        ground_workspace_from_json(encoded, max_bytes=10)
    with pytest.raises(ValueError, match="point limit"):
        ground_workspace_from_json(encoded, max_points=1)


def test_result_and_csv_exports_are_lossless_and_stable() -> None:
    result = _workspace().result

    assert ground_result_json(result) == result.to_json()
    assert load_ground_result_json(ground_result_json(result)) == result

    trajectory = ground_trajectory_csv(result)
    assert trajectory.endswith("\n")
    assert "\r" not in trajectory
    trajectory_rows = list(csv.reader(io.StringIO(trajectory)))
    assert trajectory_rows[0] == (
        "sample_index,time_s,phase,frame,position_x_m,position_y_m,position_z_m,"
        "velocity_x_m_s,velocity_y_m_s,velocity_z_m_s,"
        "angular_velocity_x_rad_s,angular_velocity_y_rad_s,angular_velocity_z_rad_s"
    ).split(",")
    assert len(trajectory_rows) == len(result.trajectory) + 1
    assert trajectory_rows[1][7] == "0.976"
    assert trajectory_rows[1][12] == "-2.81030444965"

    events = ground_event_csv(result)
    assert events.endswith("\n")
    assert "\r" not in events
    event_rows = list(csv.reader(io.StringIO(events)))
    assert event_rows[0][:5] == [
        "sequence",
        "event_type",
        "time_s",
        "frame",
        "position_x_m",
    ]
    assert len(event_rows) == len(result.events) + 1
    assert event_rows[1][7] == "1"
    assert event_rows[1][10] == "0.976"


def test_comparison_uses_one_absolute_window_without_extrapolating() -> None:
    primary = GroundPlaybackTimeline(
        load_ground_result_json(json.dumps(_result_payload()))
    )
    comparison = GroundPlaybackTimeline(
        load_ground_result_json(json.dumps(_comparison_payload()))
    )
    session = GroundPlaybackComparison(primary, comparison)

    assert session.start_time_s == pytest.approx(primary.start_time_s)
    assert session.end_time_s == pytest.approx(comparison.end_time_s)
    start = session.frame_at(primary.start_time_s)
    assert start.primary_state == "active"
    assert start.comparison_state == "waiting for first contact"
    assert start.comparison.time_s == pytest.approx(comparison.start_time_s)

    after_primary = session.frame_at(primary.end_time_s + 0.1)
    assert after_primary.primary_state == "held at rest"
    assert after_primary.comparison_state == "active"
    assert after_primary.primary.time_s == pytest.approx(primary.end_time_s)


def test_comparison_table_and_exports_are_complete_and_deterministic() -> None:
    session = GroundPlaybackComparison(
        GroundPlaybackTimeline(load_ground_result_json(json.dumps(_result_payload()))),
        GroundPlaybackTimeline(
            load_ground_result_json(json.dumps(_comparison_payload()))
        ),
    )

    rows = session.metric_rows
    assert {row.metric_id for row in rows} == {
        "carry_distance_m",
        "bounce_air_distance_m",
        "skid_distance_m",
        "roll_distance_m",
        "surface_path_distance_m",
        "total_distance_m",
        "final_downrange_m",
        "final_offline_m",
        "bounce_count",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "event_count",
        "trajectory_sample_count",
    }
    assert next(
        row for row in rows if row.metric_id == "start_time_s"
    ).delta == pytest.approx(0.2)
    assert session.provenance_rows[0].field == "Request ID"
    assert session.provenance_rows[0].primary == "surface-run-analytic"
    assert session.provenance_rows[0].comparison == "comparison-run"

    encoded = ground_comparison_json(session)
    assert encoded == ground_comparison_json(session)
    document = json.loads(encoded)
    assert document["schema_version"] == "rate-of-closure-ground-playback-comparison/v1"
    assert document["delta_definition"] == "comparison_minus_primary"
    assert document["primary"]["request_id"] == "surface-run-analytic"
    assert document["comparison"]["request_id"] == "comparison-run"
    csv_text = ground_comparison_csv(session)
    assert csv_text.endswith("\n")
    assert (
        "metric_id,label,unit,primary,comparison,comparison_minus_primary" in csv_text
    )
