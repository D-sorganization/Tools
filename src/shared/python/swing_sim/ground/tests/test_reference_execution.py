"""End-to-end contract tests for the bounded ground reference executor."""

from __future__ import annotations

from dataclasses import replace

import pytest

import shared.python.swing_sim.ground.reference_execution as reference_execution
from shared.python.swing_sim.ground import (
    GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION,
    BounceTermination,
    BounceTerminationReason,
    GroundCompositionError,
    GroundReferenceCancelled,
    GroundReferenceExecution,
    GroundReferenceExecutionError,
    GroundReferencePhase,
    GroundResultStatus,
    GroundSimulationRequest,
    GroundTerminationReason,
    PlanarSurfaceDomain,
    SkidRollSettings,
    SkidRollTerminationReason,
    SurfaceResolver,
    run_ground_reference,
    simulate_repeated_bounce,
    simulate_skid_roll,
)

from ..request_identity import ground_request_fingerprint
from ._support import _request, _settled_prefix, _surface, _surface_run_request


def _rest_request() -> GroundSimulationRequest:
    surface = replace(
        _surface(),
        normal_restitution=0.2,
        static_friction=0.3,
        kinetic_friction=0.2,
        rolling_resistance=0.2,
    )
    request = _surface_run_request(surface=surface, max_time_s=4.0)
    return replace(
        request,
        output_interval_s=0.1,
        last_separated_state=replace(
            request.last_separated_state,
            velocity_m_s=(1.0, -0.1, 0.0),
        ),
        first_penetrating_state=replace(
            request.first_penetrating_state,
            velocity_m_s=(1.0, -0.1, 0.0),
        ),
    )


def test_reference_executor_is_deterministic_and_returns_complete_rest() -> None:
    request = _rest_request()

    first = run_ground_reference(request)
    second = run_ground_reference(request)

    assert GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION == "ground-reference-execution/v1"
    assert first.to_json() == second.to_json()
    assert first.status is GroundResultStatus.COMPLETE
    assert first.termination.reason is GroundTerminationReason.REST
    assert first.summary is not None
    assert first.summary.total_distance_m > first.summary.carry_distance_m


def test_reference_executor_preserves_representable_censored_time_limit() -> None:
    request = _surface_run_request(max_time_s=0.2)

    result = run_ground_reference(request)

    assert result.status is GroundResultStatus.PARTIAL
    assert result.termination.reason is GroundTerminationReason.TIME_LIMIT
    assert result.summary is not None


def test_reference_executor_preserves_representable_event_limit() -> None:
    request = _rest_request()
    request = replace(
        request,
        max_events=1,
        surface=replace(request.surface, normal_restitution=0.0),
    )

    result = run_ground_reference(request)

    assert result.status is GroundResultStatus.PARTIAL
    assert result.termination.reason is GroundTerminationReason.EVENT_LIMIT
    assert result.summary is not None


def test_reference_executor_preserves_representable_left_surface() -> None:
    request = _rest_request()
    resolver = SurfaceResolver(
        PlanarSurfaceDomain(
            request.surface,
            lower_coordinate_m=-1.0,
            upper_coordinate_m=0.05,
        )
    )

    result = run_ground_reference(
        request,
        GroundReferenceExecution(resolver=resolver),
    )

    assert result.status is GroundResultStatus.COMPLETE
    assert result.termination.reason is GroundTerminationReason.LEFT_SURFACE


def test_bounce_cancellation_is_typed_and_retains_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _surface_run_request()
    prefix = replace(
        _settled_prefix(request),
        handoff_state=None,
        termination=BounceTermination(
            BounceTerminationReason.CANCELLED,
            request.last_separated_state.time_s,
            0.0,
        ),
    )
    monkeypatch.setattr(
        reference_execution,
        "simulate_repeated_bounce",
        lambda *_args, **_kwargs: prefix,
    )

    with pytest.raises(GroundReferenceCancelled) as caught:
        run_ground_reference(request)

    assert caught.value.phase is GroundReferencePhase.BOUNCE
    assert caught.value.native_reason == "cancelled"
    assert caught.value.request_fingerprint_sha256 == ground_request_fingerprint(
        request
    )


def test_noncomposable_bounce_terminal_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    time_limited = simulate_repeated_bounce(
        replace(request, max_time_s=0.05, output_interval_s=0.01)
    )
    event_limited = simulate_repeated_bounce(replace(request, max_events=1))
    prefixes = (
        time_limited,
        event_limited,
        replace(
            event_limited,
            termination=replace(
                event_limited.termination,
                reason=BounceTerminationReason.NO_RECONTACT,
            ),
        ),
        replace(
            event_limited,
            termination=replace(
                event_limited.termination,
                reason=BounceTerminationReason.NUMERICAL_FAILURE,
            ),
        ),
    )

    for prefix in prefixes:
        monkeypatch.setattr(
            reference_execution,
            "simulate_repeated_bounce",
            lambda *_args, _prefix=prefix, **_kwargs: _prefix,
        )
        with pytest.raises(GroundReferenceExecutionError) as caught:
            run_ground_reference(request)

        assert caught.value.phase is GroundReferencePhase.BOUNCE
        assert caught.value.native_reason == prefix.termination.reason.value
        assert caught.value.request_fingerprint_sha256 == ground_request_fingerprint(
            request
        )


def test_noncomposable_skid_step_limit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _surface_run_request()
    prefix = _settled_prefix(request)
    monkeypatch.setattr(
        reference_execution,
        "simulate_repeated_bounce",
        lambda *_args, **_kwargs: prefix,
    )
    execution = GroundReferenceExecution(
        skid_roll_settings=SkidRollSettings(max_steps=1)
    )

    with pytest.raises(GroundReferenceExecutionError) as caught:
        run_ground_reference(request, execution)

    assert caught.value.phase is GroundReferencePhase.SKID_ROLL
    assert caught.value.native_reason == "step_limit"


def test_all_noncomposable_skid_terminals_stop_before_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _surface_run_request(max_time_s=0.2)
    prefix = _settled_prefix(request)
    suffix = simulate_skid_roll(request, prefix)
    monkeypatch.setattr(
        reference_execution,
        "simulate_repeated_bounce",
        lambda *_args, **_kwargs: prefix,
    )

    def forbidden_compose(*_args: object) -> object:
        raise AssertionError("composition must not run for a noncomposable suffix")

    monkeypatch.setattr(reference_execution, "compose_ground_result", forbidden_compose)
    reasons = (
        SkidRollTerminationReason.CANCELLED,
        SkidRollTerminationReason.STEP_LIMIT,
        SkidRollTerminationReason.UNSUPPORTED_SURFACE,
        SkidRollTerminationReason.NUMERICAL_FAILURE,
    )
    for reason in reasons:
        native = replace(
            suffix,
            termination=replace(suffix.termination, reason=reason),
        )
        monkeypatch.setattr(
            reference_execution,
            "simulate_skid_roll",
            lambda *_args, _native=native: _native,
        )
        expected = (
            GroundReferenceCancelled
            if reason is SkidRollTerminationReason.CANCELLED
            else GroundReferenceExecutionError
        )

        with pytest.raises(expected) as caught:
            run_ground_reference(request)

        assert caught.value.phase is GroundReferencePhase.SKID_ROLL
        assert caught.value.native_reason == reason.value


def test_zero_duration_rest_is_a_typed_composition_failure() -> None:
    request = _surface_run_request()
    request = replace(
        request,
        surface=replace(request.surface, normal_restitution=0.0),
        last_separated_state=replace(
            request.last_separated_state,
            velocity_m_s=(0.0, -0.1, 0.0),
            angular_velocity_rad_s=(0.0, 0.0, 0.0),
        ),
        first_penetrating_state=replace(
            request.first_penetrating_state,
            velocity_m_s=(0.0, -0.1, 0.0),
            angular_velocity_rad_s=(0.0, 0.0, 0.0),
        ),
    )

    with pytest.raises(GroundReferenceExecutionError) as caught:
        run_ground_reference(request)

    assert caught.value.phase is GroundReferencePhase.COMPOSITION
    assert caught.value.native_reason == "composition_error"
    assert isinstance(caught.value.__cause__, GroundCompositionError)


def test_preflight_cancellation_is_one_bounce_probe() -> None:
    calls = 0

    def cancel_preflight() -> bool:
        nonlocal calls
        calls += 1
        return True

    with pytest.raises(GroundReferenceCancelled) as preflight:
        run_ground_reference(
            _surface_run_request(),
            GroundReferenceExecution(is_cancelled=cancel_preflight),
        )
    assert preflight.value.phase is GroundReferencePhase.BOUNCE
    assert calls == 1


def test_mid_bounce_cancellation_is_two_bounce_probes() -> None:
    calls = 0

    def cancel_mid_bounce() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(GroundReferenceCancelled) as mid_bounce:
        run_ground_reference(
            _surface_run_request(),
            GroundReferenceExecution(is_cancelled=cancel_mid_bounce),
        )
    assert mid_bounce.value.phase is GroundReferencePhase.BOUNCE
    assert calls == 2


def test_skid_cancellation_reuses_hook_after_one_bounce_probe() -> None:
    calls = 0

    def cancel_in_skid() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    request = _surface_run_request()
    request = replace(
        request,
        surface=replace(request.surface, normal_restitution=0.0),
    )
    with pytest.raises(GroundReferenceCancelled) as skid:
        run_ground_reference(
            request,
            GroundReferenceExecution(is_cancelled=cancel_in_skid),
        )
    assert skid.value.phase is GroundReferencePhase.SKID_ROLL
    assert calls == 2


def test_same_cancellation_hook_reaches_both_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _rest_request()
    prefix = _settled_prefix(request)
    suffix = simulate_skid_roll(request, prefix)
    observed: dict[str, object] = {}

    def fake_bounce(*_args: object, **kwargs: object) -> object:
        observed["bounce"] = kwargs["is_cancelled"]
        return prefix

    def fake_skid(*_args: object) -> object:
        selected = _args[2]
        observed["skid"] = selected.is_cancelled
        return suffix

    def callback() -> bool:
        return False

    monkeypatch.setattr(reference_execution, "simulate_repeated_bounce", fake_bounce)
    monkeypatch.setattr(reference_execution, "simulate_skid_roll", fake_skid)

    result = run_ground_reference(
        request,
        GroundReferenceExecution(is_cancelled=callback),
    )

    assert result.termination.reason is GroundTerminationReason.REST
    assert observed == {"bounce": callback, "skid": callback}


@pytest.mark.parametrize(
    "execution",
    [
        object(),
        GroundReferenceExecution.__new__(GroundReferenceExecution),
    ],
)
def test_reference_executor_rejects_nonexact_execution(execution: object) -> None:
    with pytest.raises(ValueError, match="exact GroundReferenceExecution"):
        run_ground_reference(_surface_run_request(), execution)  # type: ignore[arg-type]


def test_reference_execution_rejects_invalid_nested_controls() -> None:
    with pytest.raises(ValueError, match="bounce_settings must be exact"):
        GroundReferenceExecution(bounce_settings=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="skid_roll_settings must be exact"):
        GroundReferenceExecution(skid_roll_settings=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resolver must be exact"):
        GroundReferenceExecution(resolver=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is_cancelled must be callable"):
        GroundReferenceExecution(is_cancelled=object())  # type: ignore[arg-type]
