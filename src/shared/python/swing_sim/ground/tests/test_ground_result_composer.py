"""Composition and lossy-adapter tests for qualified final ground results."""

from __future__ import annotations

from dataclasses import replace

import pytest

from shared.python.swing_sim.ground import (
    GroundCompositionError,
    GroundEventType,
    GroundPhase,
    GroundResultStatus,
    GroundTerminationReason,
    PlanarSurfaceDomain,
    SkidRollExecution,
    SurfaceResolver,
    compose_ground_result,
    simulate_skid_roll,
    to_ground_model_result,
)
from shared.python.swing_sim.ground.bounce_types import (
    BOUNCE_HANDOFF_NOTICE,
    BOUNCE_MATERIAL_LIMITATION,
)

from ._support import _settled_prefix, _surface, _surface_run_request


def test_composer_preserves_unique_handoff_sequences_and_distance_definitions() -> None:
    surface = replace(_surface(), rolling_resistance=0.05)
    request = _surface_run_request(surface=surface)
    prefix = _settled_prefix(request)
    suffix = simulate_skid_roll(request, prefix)
    result = compose_ground_result(request, prefix, suffix)
    handoff_time = prefix.handoff_state.time_s if prefix.handoff_state else -1.0

    assert result.status is GroundResultStatus.COMPLETE
    assert result.termination.reason is GroundTerminationReason.REST
    assert (
        sum(
            event.event_type is GroundEventType.FIRST_CONTACT for event in result.events
        )
        == 1
    )
    assert tuple(event.sequence for event in result.events) == tuple(
        range(len(result.events))
    )
    assert sum(point.time_s == handoff_time for point in result.trajectory) == 1
    assert (
        next(point for point in result.trajectory if point.time_s == handoff_time).phase
        is GroundPhase.SKID
    )
    assert result.summary is not None
    assert result.summary.bounce_air_distance_m == pytest.approx(
        prefix.bounce_air_distance_m
    )
    assert result.summary.bounce_count == 1
    assert result.summary.surface_path_distance_m == pytest.approx(
        result.summary.skid_distance_m + result.summary.roll_distance_m
    )
    assert result.model_id == f"{prefix.model_id}+{suffix.model_id}"
    assert result.model_version == (f"{prefix.model_version}+{suffix.model_version}")
    assert all(
        warning in tuple(item.message for item in result.warnings)
        for warning in prefix.warnings
    )
    final = result.trajectory[-1].position_m
    assert result.summary.total_distance_m == pytest.approx(
        (final[0] ** 2 + final[2] ** 2) ** 0.5
    )


def test_immediate_capture_relabels_exact_state_without_duplicate_or_epsilon() -> None:
    surface = replace(_surface(), rolling_resistance=0.05)
    request = _surface_run_request(surface=surface)
    prefix = _settled_prefix(request, immediate=True)
    suffix = simulate_skid_roll(request, prefix)
    result = compose_ground_result(request, prefix, suffix)
    original = prefix.trajectory[0]
    transformed = result.trajectory[0]

    assert transformed.phase is GroundPhase.IMPACT
    assert transformed.time_s == original.time_s
    assert transformed.position_m == original.position_m
    assert transformed.velocity_m_s == original.velocity_m_s
    assert transformed.angular_velocity_rad_s == original.angular_velocity_rad_s
    assert sum(point.time_s == original.time_s for point in result.trajectory) == 1
    assert all(point.time_s > original.time_s for point in result.trajectory[1:])


def test_zero_duration_immediate_rest_fails_closed_without_fabricated_time() -> None:
    request = _surface_run_request()
    prefix = _settled_prefix(
        request,
        velocity_m_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
        immediate=True,
    )
    suffix = simulate_skid_roll(request, prefix)

    with pytest.raises(GroundCompositionError, match="zero-duration rest"):
        compose_ground_result(request, prefix, suffix)


def test_left_surface_summary_is_censored_and_legacy_adapter_rejects_it() -> None:
    surface = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=surface, max_time_s=1.0)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -2.0 / request.ball_radius_m),
    )
    handoff = prefix.handoff_state
    if handoff is None:
        raise RuntimeError("test prefix must expose a handoff")
    resolver = SurfaceResolver(
        PlanarSurfaceDomain(
            surface,
            lower_coordinate_m=handoff.position_m[0] - 1.0,
            upper_coordinate_m=handoff.position_m[0] + 0.5,
        )
    )
    suffix = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(resolver=resolver),
    )
    result = compose_ground_result(request, prefix, suffix)

    assert result.status is GroundResultStatus.COMPLETE
    assert result.termination.reason is GroundTerminationReason.LEFT_SURFACE
    assert result.summary is not None
    assert any(warning.code == "CENSORED_ENDPOINT" for warning in result.warnings)
    with pytest.raises(ValueError, match="rest-terminated"):
        to_ground_model_result(result)


def test_time_limit_summary_is_censored_endpoint_diagnostic() -> None:
    surface = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=surface, max_time_s=0.2)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -2.0 / request.ball_radius_m),
    )
    suffix = simulate_skid_roll(request, prefix)
    result = compose_ground_result(request, prefix, suffix)

    assert result.status is GroundResultStatus.PARTIAL
    assert result.termination.reason is GroundTerminationReason.TIME_LIMIT
    assert result.summary is not None
    endpoint = result.trajectory[-1].position_m
    assert result.summary.total_distance_m == pytest.approx(
        (endpoint[0] ** 2 + endpoint[2] ** 2) ** 0.5
    )
    assert any(warning.code == "CENSORED_ENDPOINT" for warning in result.warnings)
    with pytest.raises(ValueError, match="rest-terminated"):
        to_ground_model_result(result)


def test_internal_cancel_and_step_limit_cannot_be_serialized_as_v1_results() -> None:
    request = _surface_run_request()
    prefix = _settled_prefix(request)
    cancelled = simulate_skid_roll(
        request,
        prefix,
        SkidRollExecution(is_cancelled=lambda: True),
    )

    with pytest.raises(GroundCompositionError, match="not representable"):
        compose_ground_result(request, prefix, cancelled)


def test_suffix_rejects_final_state_that_disagrees_with_terminal_point() -> None:
    surface = replace(_surface(), rolling_resistance=0.0)
    request = _surface_run_request(surface=surface, max_time_s=0.2)
    prefix = _settled_prefix(request)
    suffix = simulate_skid_roll(request, prefix)
    impossible = replace(
        suffix.final_state,
        position_m=(999.0, suffix.final_state.position_m[1], 999.0),
    )

    with pytest.raises(ValueError, match="final state must match"):
        replace(suffix, final_state=impossible)


def test_composer_rejects_phase_results_from_a_different_request() -> None:
    request = _surface_run_request()
    prefix = _settled_prefix(request)
    suffix = simulate_skid_roll(request, prefix)
    changed = replace(request, ball_mass_kg=request.ball_mass_kg + 0.001)

    with pytest.raises(GroundCompositionError, match="request fingerprints"):
        compose_ground_result(changed, prefix, suffix)


def test_composer_preserves_material_limit_but_drops_fulfilled_handoff_notice() -> None:
    request = _surface_run_request()
    prefix = replace(
        _settled_prefix(request),
        warnings=(BOUNCE_MATERIAL_LIMITATION, BOUNCE_HANDOFF_NOTICE),
    )
    suffix = simulate_skid_roll(request, prefix)

    result = compose_ground_result(request, prefix, suffix)
    messages = tuple(warning.message for warning in result.warnings)

    assert BOUNCE_MATERIAL_LIMITATION in messages
    assert BOUNCE_HANDOFF_NOTICE not in messages
