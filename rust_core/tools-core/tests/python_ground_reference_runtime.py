"""Real-wheel contract checks for compiled ground reference execution."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import tools_core

FIXTURE = (
    Path(__file__).parents[3]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "ground_reference_pipeline_golden_v1.json"
)


def fixture_parts() -> tuple[str, str, str]:
    """Return strict request, execution, and expected result digest."""
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    execution = dict(fixture["execution"])
    execution["schema_version"] = fixture["execution_schema_version"]
    return (
        json.dumps(fixture["request"], separators=(",", ":")),
        json.dumps(execution, separators=(",", ":")),
        str(fixture["result_sha256"]),
    )


def main() -> None:
    """Exercise actual physics and cancellation through the installed wheel."""
    request_json, execution_json, expected_digest = fixture_parts()
    actual = tools_core.run_flight_to_ground_reference_v1(request_json, execution_json)
    assert hashlib.sha256(actual.encode()).hexdigest() == expected_digest
    assert tools_core.run_flight_to_ground_reference_v1(request_json) == actual
    for _ in range(100):
        assert (
            tools_core.run_flight_to_ground_reference_v1(request_json, execution_json)
            == actual
        )

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pathological = dict(fixture["request"])
    pathological["output_interval_s"] = 1e-11
    callback_calls = 0

    def counted_callback() -> bool:
        nonlocal callback_calls
        callback_calls += 1
        return False

    try:
        tools_core.run_flight_to_ground_reference_v1(
            json.dumps(pathological, separators=(",", ":")),
            execution_json,
            counted_callback,
        )
    except RuntimeError as error:
        payload = json.loads(str(error))
        assert payload["native_reason"] == "output_point_limit"
        assert callback_calls == 0
    else:
        raise AssertionError("compiled wheel accepted an unbounded output schedule")

    for restitution in (fixture["request"]["surface"]["normal_restitution"], 0.0):
        unrepresentable = json.loads(request_json)
        unrepresentable["last_separated_state"]["time_s"] = 9_000_000_000_000_000
        unrepresentable["first_penetrating_state"]["time_s"] = 9_000_000_000_000_002
        unrepresentable["surface"]["normal_restitution"] = restitution
        unrepresentable["max_time_s"] = 0.1
        unrepresentable["output_interval_s"] = 0.001
        resolution_callback_calls = 0

        def resolution_callback() -> bool:
            nonlocal resolution_callback_calls
            resolution_callback_calls += 1
            return False

        try:
            tools_core.run_flight_to_ground_reference_v1(
                json.dumps(unrepresentable, separators=(",", ":")),
                execution_json,
                resolution_callback,
            )
        except RuntimeError as error:
            payload = json.loads(str(error))
            assert payload["native_reason"] == "time_resolution"
            assert payload["phase"] == "bounce"
            assert resolution_callback_calls == 0
        else:
            raise AssertionError("compiled wheel emitted an unrepresentable time grid")

    capped = json.loads(request_json)
    capped["max_time_s"] = 200_001
    capped["output_interval_s"] = 1
    untrusted_execution = json.loads(execution_json)
    untrusted_execution["skid_roll_settings"]["max_steps"] = 9_007_199_254_740_991
    cap_callback_calls = 0

    def cap_callback() -> bool:
        nonlocal cap_callback_calls
        cap_callback_calls += 1
        return False

    assert_runtime_failure(
        capped,
        json.dumps(untrusted_execution, separators=(",", ":")),
        "execution_failure",
        "bounce",
        "output_point_limit",
        cap_callback,
    )
    assert cap_callback_calls == 0

    bounded_steps = json.loads(request_json)
    bounded_steps["max_time_s"] = 1.0
    bounded_steps["output_interval_s"] = 1.0
    oversized_steps = json.loads(execution_json)
    oversized_steps["skid_roll_settings"]["integration_step_s"] = 1e-11
    oversized_steps["skid_roll_settings"]["max_steps"] = 1_000_001
    step_callback_calls = 0

    def step_callback() -> bool:
        nonlocal step_callback_calls
        step_callback_calls += 1
        return False

    assert_runtime_failure(
        bounded_steps,
        json.dumps(oversized_steps, separators=(",", ":")),
        "execution_failure",
        "skid_roll",
        "integration_step_limit",
        step_callback,
    )
    assert step_callback_calls == 0

    oversized_events = json.loads(request_json)
    oversized_events["max_events"] = 10_001
    assert_runtime_failure(
        oversized_events,
        execution_json,
        "execution_failure",
        "bounce",
        "event_count_limit",
    )

    bounce_overflow = json.loads(request_json)
    bounce_overflow["ball_radius_m"] = 1_000_000
    bounce_overflow["surface"].update(
        normal_restitution=1, static_friction=5, kinetic_friction=5
    )
    for state_name, height in (
        ("last_separated_state", 1_000_001),
        ("first_penetrating_state", 999_999),
    ):
        state = bounce_overflow[state_name]
        state["position_m"][1] = height
        state["velocity_m_s"] = [0, -9_000_000_000_000_000, 0]
        state["angular_velocity_rad_s"] = [0, 0, 9_000_000_000_000_000]
    assert_runtime_failure(
        bounce_overflow,
        execution_json,
        "numerical_failure",
        "bounce",
        "numeric_range",
    )

    surface_overflow = json.loads(request_json)
    surface_overflow["surface"]["normal_restitution"] = 0
    for state_name in ("last_separated_state", "first_penetrating_state"):
        surface_overflow[state_name]["position_m"][0] = 9_000_000_000_000_000
        surface_overflow[state_name]["velocity_m_s"][0] = 9_000_000_000_000_000
    surface_overflow["max_time_s"] = 0.01
    surface_overflow["output_interval_s"] = 0.001
    assert_runtime_failure(
        surface_overflow,
        execution_json,
        "numerical_failure",
        "skid_roll",
        "numeric_range",
    )

    composition_overflow = json.loads(request_json)
    for state_name in ("last_separated_state", "first_penetrating_state"):
        composition_overflow[state_name]["position_m"][0] = 6_500_000_000_000_000
        composition_overflow[state_name]["position_m"][2] = 6_500_000_000_000_000
    assert_runtime_failure(
        composition_overflow,
        execution_json,
        "numerical_failure",
        "composition",
        "numeric_range",
    )

    for restitution in (fixture["request"]["surface"]["normal_restitution"], 0.0):
        representable = json.loads(request_json)
        representable["last_separated_state"]["time_s"] = 1_000_000_000_000
        representable["first_penetrating_state"]["time_s"] = 1_000_000_000_002
        representable["surface"]["normal_restitution"] = restitution
        representable["output_interval_s"] = 0.00125
        result = json.loads(
            tools_core.run_flight_to_ground_reference_v1(
                json.dumps(representable, separators=(",", ":")), execution_json
            )
        )
        times = [point["time_s"] for point in result["trajectory"]]
        assert all(left < right for left, right in zip(times, times[1:], strict=False))

    event_limited = json.loads(request_json)
    event_limited["max_events"] = 1
    rolling_speed = 1.0
    rolling_spin = -rolling_speed / event_limited["ball_radius_m"]
    for state_name in ("last_separated_state", "first_penetrating_state"):
        event_limited[state_name]["velocity_m_s"] = [rolling_speed, -0.04, 0]
        event_limited[state_name]["angular_velocity_rad_s"] = [0, 0, rolling_spin]
    event_result = json.loads(
        tools_core.run_flight_to_ground_reference_v1(
            json.dumps(event_limited, separators=(",", ":")), execution_json
        )
    )
    assert event_result["status"] == "partial"
    assert event_result["termination"]["reason"] == "event_limit"
    assert event_result["termination"]["completed"] is False
    assert len(event_result["events"]) == 1
    assert (
        event_result["trajectory"][-1]["time_s"]
        == event_result["termination"]["time_s"]
    )

    rebound_event_limited = json.loads(request_json)
    rebound_event_limited["max_events"] = 1
    assert_runtime_failure(
        rebound_event_limited,
        execution_json,
        "execution_failure",
        "bounce",
        "event_limit",
    )

    try:
        tools_core.run_flight_to_ground_reference_v1(
            request_json, execution_json, lambda: True
        )
    except InterruptedError as error:
        payload = json.loads(str(error))
        assert payload["code"] == "cancelled"
        assert payload["phase"] == "bounce"
    else:
        raise AssertionError("compiled wheel did not surface typed cancellation")

    def raises() -> bool:
        raise RuntimeError("callback-sentinel")

    try:
        tools_core.run_flight_to_ground_reference_v1(
            request_json, execution_json, raises
        )
    except RuntimeError as error:
        assert str(error) == "callback-sentinel"
    else:
        raise AssertionError("callback exception was swallowed")

    try:
        tools_core.run_flight_to_ground_reference_v1(
            request_json, execution_json, lambda: 1
        )
    except TypeError:
        pass
    else:
        raise AssertionError("non-boolean callback result was accepted")

    concurrent = json.loads(request_json)
    concurrent["max_time_s"] = 100.0
    concurrent["output_interval_s"] = 1.0
    concurrent["surface"]["surface_velocity_m_s"] = [0.1, 0.0, 0.0]
    callback_started = threading.Event()
    cancel_requested = threading.Event()
    outcome: list[BaseException | str] = []

    def cancellation_poll() -> bool:
        callback_started.set()
        return cancel_requested.is_set()

    def run_concurrently() -> None:
        try:
            tools_core.run_flight_to_ground_reference_v1(
                json.dumps(concurrent, separators=(",", ":")),
                execution_json,
                cancellation_poll,
            )
            outcome.append("completed")
        except BaseException as error:  # noqa: BLE001 - preserve binding evidence
            outcome.append(error)

    worker = threading.Thread(target=run_concurrently, daemon=True)
    previous_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(1_000.0)
        worker.start()
        assert callback_started.wait(timeout=2.0)
        started = time.perf_counter()
        cancel_requested.set()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert time.perf_counter() - started < 1.0
        assert len(outcome) == 1 and isinstance(outcome[0], InterruptedError)
    finally:
        sys.setswitchinterval(previous_interval)


def assert_runtime_failure(
    request: dict[str, object],
    execution_json: str,
    code: str,
    phase: str,
    reason: str,
    callback: object | None = None,
) -> None:
    """Assert one typed compiled runtime failure without a panic escape."""
    try:
        tools_core.run_flight_to_ground_reference_v1(
            json.dumps(request, separators=(",", ":")), execution_json, callback
        )
    except RuntimeError as error:
        payload = json.loads(str(error))
        assert payload["code"] == code
        assert payload["phase"] == phase
        assert payload["native_reason"] == reason
    else:
        raise AssertionError(f"compiled wheel did not reject {reason}")


if __name__ == "__main__":
    main()
