"""Phase-aware playback contracts for imported ground results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.simulation.ground_playback import (
    GroundPlaybackTimeline,
    load_ground_result_json,
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
