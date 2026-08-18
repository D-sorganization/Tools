"""Phase-aware playback contracts for ground and regional results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.simulation.ground_playback import (
    MAX_RENDERED_PATH_POINTS,
    GroundPlaybackTimeline,
    load_ground_result_json,
    select_ground_playback_indices,
    timeline_from_regional_execution,
)
from shared.python.swing_sim.ground import RegionalGroundExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURES = Path(__file__).parents[2] / "src/rate_of_closure/web/src/model/__fixtures__"


def _ground_payload() -> dict[str, object]:
    payload = json.loads((FIXTURES / "flight_to_ground_golden_v1.json").read_text())
    return payload["result"]


def _regional_payload(name: str = "representable") -> dict[str, object]:
    payload = json.loads(
        (FIXTURES / "ground_regional_execution_golden_v1.json").read_text()
    )
    return payload[name]["result"]


def test_strict_import_builds_absolute_phase_safe_timeline() -> None:
    timeline = GroundPlaybackTimeline(
        load_ground_result_json(json.dumps(_ground_payload()))
    )
    assert timeline.start_time_s > 0.0
    assert timeline.duration_s > 0.0
    assert timeline.phase_time("skid") is not None
    assert timeline.carry_position_m == timeline.result.trajectory[0].position_m
    assert timeline.endpoint_position_m == timeline.result.trajectory[-1].position_m

    transition_time = timeline.phase_time("roll")
    assert transition_time is not None
    before = timeline.frame_at(transition_time - 1e-9)
    at = timeline.frame_at(transition_time)
    assert before.phase != at.phase
    assert before.interpolation_fraction == 0.0


def test_timeline_steps_clamps_and_labels_observed_end_honestly() -> None:
    timeline = GroundPlaybackTimeline(
        load_ground_result_json(json.dumps(_ground_payload()))
    )
    assert timeline.step_time(timeline.start_time_s, -1) == timeline.start_time_s
    assert timeline.step_time(timeline.end_time_s, 1) == timeline.end_time_s
    assert (
        timeline.frame_at(timeline.start_time_s - 100).time_s == timeline.start_time_s
    )
    assert timeline.frame_at(timeline.end_time_s + 100).time_s == timeline.end_time_s
    assert timeline.end_label in {"Rest", "End / left surface"}

    payload = _ground_payload()
    payload["status"] = "partial"
    payload["trajectory"] = payload["trajectory"][:-1]  # type: ignore[index]
    payload["events"] = payload["events"][:-1]  # type: ignore[index]
    final = payload["trajectory"][-1]  # type: ignore[index]
    position = final["position_m"]
    summary = payload["summary"]
    summary["final_downrange_m"] = position[0]  # type: ignore[index]
    summary["final_offline_m"] = position[2]  # type: ignore[index]
    summary["total_distance_m"] = (
        position[0] ** 2 + position[2] ** 2  # type: ignore[operator,index]
    ) ** 0.5
    payload["termination"] = {
        "completed": False,
        "reason": "time_limit",
        "time_s": final["time_s"],
    }
    partial = GroundPlaybackTimeline(load_ground_result_json(json.dumps(payload)))
    assert partial.end_label == "Observed end"


def test_regional_adapter_returns_nested_result_without_executing_physics() -> None:
    execution = RegionalGroundExecutionResult.from_dict(_regional_payload())
    timeline = timeline_from_regional_execution(execution)
    assert timeline.result is execution.ground_result


@pytest.mark.parametrize("name", ["cancelled", "failed"])
def test_regional_adapter_rejects_non_playable_envelopes(name: str) -> None:
    execution = RegionalGroundExecutionResult.from_dict(_regional_payload(name))
    with pytest.raises(ValueError, match="playable ground result"):
        timeline_from_regional_execution(execution)


def test_import_and_timeline_fail_closed() -> None:
    encoded = json.dumps(_ground_payload())
    with pytest.raises(ValueError, match="size limit"):
        load_ground_result_json(encoded, max_bytes=10)
    with pytest.raises(TypeError, match="exact RegionalGroundExecutionResult"):
        timeline_from_regional_execution(object())  # type: ignore[arg-type]


def test_large_visual_path_selection_is_bounded_and_preserves_boundaries() -> None:
    count = 100_000
    times = tuple(index * 0.001 for index in range(count))
    phases = tuple(
        "impact" if index < 10 else "skid" if index < 50_000 else "roll"
        for index in range(count)
    )
    event_times = (times[25_000], times[75_000])

    indices = select_ground_playback_indices(phases, times, event_times)

    assert len(indices) <= MAX_RENDERED_PATH_POINTS
    assert {0, count - 1, 9, 10, 49_999, 50_000, 25_000, 75_000} <= set(indices)
