"""Reusable, deterministic records for ground-contract tests."""

from __future__ import annotations

from dataclasses import replace

from shared.python.swing_sim.ground import (
    BounceAirSegment,
    BounceTermination,
    BounceTerminationReason,
    CalibrationKind,
    GroundCalibration,
    GroundContactState,
    GroundEvent,
    GroundEventType,
    GroundFrame,
    GroundPhase,
    GroundProvenance,
    GroundResultStatus,
    GroundSimulationRequest,
    GroundSimulationResult,
    GroundSummary,
    GroundSurfaceProfile,
    GroundTermination,
    GroundTerminationReason,
    GroundTrajectoryPoint,
    GroundWarning,
    GroundWarningSeverity,
    ImpactEnergyLedger,
    ImpactImpulseResult,
    ImpactRegime,
    RepeatedBounceResult,
)


def _provenance() -> GroundProvenance:
    return GroundProvenance(
        producer="tools.rate_of_closure",
        producer_version="1.0.0",
        source_revision="60ac5b46",
        input_sha256="a" * 64,
    )


def _calibration() -> GroundCalibration:
    return GroundCalibration(
        calibration_id="literature-default-2026-08",
        kind=CalibrationKind.LITERATURE,
        source="documented literature basis",
        confidence=0.6,
    )


def _surface() -> GroundSurfaceProfile:
    return GroundSurfaceProfile(
        surface_id="firm-fairway",
        provider_id="tools.planar-surface",
        provider_version="1.0.0",
        frame=GroundFrame.TARGET,
        height_m=0.0,
        normal_unit=(0.0, 1.0, 0.0),
        surface_velocity_m_s=(0.0, 0.0, 0.0),
        normal_restitution=0.42,
        static_friction=0.35,
        kinetic_friction=0.28,
        rolling_resistance=0.04,
        firmness_pa=1_200_000.0,
        hardness_fraction=0.7,
        grass_height_m=0.012,
        compressibility_fraction=0.2,
        compression_damping_fraction=0.25,
        turf_density_kg_m3=180.0,
        moisture_fraction=0.3,
    )


def _contact() -> GroundContactState:
    return GroundContactState(
        time_s=5.19,
        frame=GroundFrame.TARGET,
        position_m=(209.7, 0.024, -3.01),
        velocity_m_s=(31.0, -12.0, 1.5),
        angular_velocity_rad_s=(0.0, 260.0, -4.0),
    )


def _penetrating_contact() -> GroundContactState:
    return replace(_contact(), time_s=5.2, position_m=(210.0, 0.019, -3.0))


def _request() -> GroundSimulationRequest:
    return GroundSimulationRequest(
        request_id="ground-run-001",
        surface=_surface(),
        last_separated_state=_contact(),
        first_penetrating_state=_penetrating_contact(),
        ball_radius_m=0.02135,
        ball_mass_kg=0.04593,
        rotational_inertia_factor=0.4,
        max_time_s=12.0,
        output_interval_s=0.01,
        max_events=64,
        calibration=_calibration(),
        provenance=_provenance(),
    )


def _point(
    time_s: float,
    position_m: tuple[float, float, float],
    phase: GroundPhase,
) -> GroundTrajectoryPoint:
    stopped = phase is GroundPhase.REST
    return GroundTrajectoryPoint(
        time_s=time_s,
        frame=GroundFrame.TARGET,
        position_m=position_m,
        velocity_m_s=(0.0, 0.0, 0.0) if stopped else (15.0, 0.0, 0.5),
        angular_velocity_rad_s=(0.0, 0.0, 0.0) if stopped else (0.0, 200.0, -2.0),
        phase=phase,
    )


def _trajectory() -> tuple[GroundTrajectoryPoint, ...]:
    return (
        replace(
            _point(5.2, (210.0, 0.02135, -3.0), GroundPhase.IMPACT),
            velocity_m_s=(24.0, 6.0, 1.0),
        ),
        _point(5.5, (218.0, 0.02135, -2.8), GroundPhase.SKID),
        _point(6.0, (224.0, 0.02135, -2.5), GroundPhase.ROLL),
        _point(8.0, (228.0, 0.02135, -2.25), GroundPhase.REST),
    )


def _event(
    sequence: int,
    event_type: GroundEventType,
    point: GroundTrajectoryPoint,
    velocity_before_m_s: tuple[float, float, float],
) -> GroundEvent:
    return GroundEvent(
        sequence=sequence,
        event_type=event_type,
        time_s=point.time_s,
        frame=GroundFrame.TARGET,
        position_m=point.position_m,
        velocity_before_m_s=velocity_before_m_s,
        velocity_after_m_s=point.velocity_m_s,
        angular_velocity_before_rad_s=(0.0, 210.0, -2.5),
        angular_velocity_after_rad_s=point.angular_velocity_rad_s,
    )


def _events(points: tuple[GroundTrajectoryPoint, ...]) -> tuple[GroundEvent, ...]:
    first = _event(0, GroundEventType.FIRST_CONTACT, points[0], (31.0, -12.0, 1.5))
    bounce = replace(
        _event(1, GroundEventType.BOUNCE, points[1], (24.0, -3.0, 1.0)),
        velocity_after_m_s=(20.0, 2.0, 0.8),
    )
    skid = replace(
        _event(2, GroundEventType.SKID_TO_ROLL, points[2], (15.0, 0.0, 0.5)),
        velocity_after_m_s=(14.8, 0.0, 0.5),
    )
    rest = _event(3, GroundEventType.REST, points[3], (0.1, 0.0, 0.0))
    return (first, bounce, skid, rest)


def _summary() -> GroundSummary:
    return GroundSummary(
        carry_distance_m=210.0214274787,
        bounce_air_distance_m=3.0,
        skid_distance_m=5.0,
        roll_distance_m=10.0,
        surface_path_distance_m=15.0,
        total_distance_m=228.0111017034039,
        final_downrange_m=228.0,
        final_offline_m=-2.25,
        bounce_count=1,
    )


def _result() -> GroundSimulationResult:
    points = _trajectory()
    return GroundSimulationResult(
        request_id="ground-run-001",
        surface_id="firm-fairway",
        frame=GroundFrame.TARGET,
        model_id="tools-ground-reference",
        model_version="0.1.0",
        status=GroundResultStatus.COMPLETE,
        trajectory=points,
        events=_events(points),
        summary=_summary(),
        termination=GroundTermination(GroundTerminationReason.REST, 8.0, True),
        calibration=_calibration(),
        warnings=(
            GroundWarning(
                code="LITERATURE_CALIBRATION",
                severity=GroundWarningSeverity.INFO,
                message="Validate against measured turf before decision use.",
            ),
        ),
        unavailable_fields=(),
        provenance=_provenance(),
    )


def _failed_result() -> GroundSimulationResult:
    return replace(
        _result(),
        status=GroundResultStatus.FAILED,
        trajectory=(),
        events=(),
        summary=None,
        termination=GroundTermination(
            GroundTerminationReason.NUMERICAL_FAILURE,
            5.2,
            False,
        ),
    )


def _surface_run_request(
    *,
    surface: GroundSurfaceProfile | None = None,
    max_time_s: float = 12.0,
    max_events: int = 64,
) -> GroundSimulationRequest:
    """Return a compact single-plane request for skid/roll analytic tests."""
    selected = _surface() if surface is None else surface
    radius = 0.02135
    normal = selected.normal_unit
    contact_position = (
        radius * normal[0],
        selected.height_m + radius * normal[1],
        radius * normal[2],
    )
    separated = GroundContactState(
        time_s=1.0,
        frame=GroundFrame.TARGET,
        position_m=tuple(
            contact_position[index] + 0.001 * normal[index] for index in range(3)
        ),
        velocity_m_s=(
            7.0 - 0.1 * normal[0],
            -0.1 * normal[1],
            -0.1 * normal[2],
        ),
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
    )
    penetrating = replace(
        separated,
        time_s=1.01,
        position_m=tuple(
            contact_position[index] - 0.001 * normal[index] for index in range(3)
        ),
    )
    return GroundSimulationRequest(
        request_id="surface-run-analytic",
        surface=selected,
        last_separated_state=separated,
        first_penetrating_state=penetrating,
        ball_radius_m=radius,
        ball_mass_kg=0.04593,
        rotational_inertia_factor=0.4,
        max_time_s=max_time_s,
        output_interval_s=0.01,
        max_events=max_events,
        calibration=_calibration(),
        provenance=_provenance(),
    )


def _impact_result(
    before: GroundContactState,
    after: GroundContactState,
    restitution: float,
) -> ImpactImpulseResult:
    """Build a structurally valid deterministic impact record for composition tests."""
    return ImpactImpulseResult(
        state_before=before,
        state_after=after,
        regime=ImpactRegime.STICKING,
        normal_impulse_n_s=0.01,
        tangential_impulse_n_s=(0.0, 0.0, 0.0),
        total_impulse_n_s=(0.0, 0.01, 0.0),
        contact_velocity_before_m_s=before.velocity_m_s,
        contact_velocity_after_m_s=after.velocity_m_s,
        effective_restitution=restitution,
        friction_utilization=0.0,
        energy=ImpactEnergyLedger(2.0, 1.0, 0.0, 1.0),
    )


def _settled_prefix(
    request: GroundSimulationRequest,
    *,
    velocity_m_s: tuple[float, float, float] = (7.0, 0.0, 0.0),
    angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    immediate: bool = False,
) -> RepeatedBounceResult:
    """Return a validated #4270 SETTLED_TO_SKID prefix with exact handoff state."""
    radius = request.ball_radius_m
    normal = request.surface.normal_unit
    first_time = 1.005
    first_position = (
        radius * normal[0],
        request.surface.height_m + radius * normal[1],
        radius * normal[2],
    )
    first_before = GroundContactState(
        first_time,
        request.surface.frame,
        first_position,
        tuple(velocity_m_s[index] - 0.1 * normal[index] for index in range(3)),
        angular_velocity_rad_s,
    )
    if immediate:
        handoff = replace(
            first_before,
            velocity_m_s=velocity_m_s,
            angular_velocity_rad_s=angular_velocity_rad_s,
        )
        impact = _impact_result(first_before, handoff, 0.0)
        event = _event_from_impact(0, GroundEventType.FIRST_CONTACT, impact)
        point = GroundTrajectoryPoint(
            handoff.time_s,
            handoff.frame,
            handoff.position_m,
            handoff.velocity_m_s,
            handoff.angular_velocity_rad_s,
            GroundPhase.SKID,
        )
        return RepeatedBounceResult(
            request.request_id,
            request.surface.surface_id,
            request.surface.frame,
            "tools-ground-impact-bounce",
            "1.0.0",
            (point,),
            (event,),
            (impact,),
            (),
            handoff,
            BounceTermination(BounceTerminationReason.SETTLED_TO_SKID, first_time, 0.0),
            ("test prefix",),
        )
    second_time = first_time + 0.02
    first_after = replace(
        first_before,
        velocity_m_s=tuple(
            velocity_m_s[index] + 0.1 * normal[index] for index in range(3)
        ),
    )
    second_position = tuple(
        first_position[index] + velocity_m_s[index] * 0.02 for index in range(3)
    )
    second_before = replace(
        first_before,
        time_s=second_time,
        position_m=second_position,
        velocity_m_s=tuple(
            velocity_m_s[index] - 0.1 * normal[index] for index in range(3)
        ),
    )
    handoff = replace(
        second_before,
        velocity_m_s=velocity_m_s,
        angular_velocity_rad_s=angular_velocity_rad_s,
    )
    impacts = (
        _impact_result(first_before, first_after, 0.1),
        _impact_result(second_before, handoff, 0.0),
    )
    events = (
        _event_from_impact(0, GroundEventType.FIRST_CONTACT, impacts[0]),
        _event_from_impact(1, GroundEventType.BOUNCE, impacts[1]),
    )
    points = (
        GroundTrajectoryPoint(
            first_time,
            request.surface.frame,
            first_position,
            first_after.velocity_m_s,
            first_after.angular_velocity_rad_s,
            GroundPhase.IMPACT,
        ),
        GroundTrajectoryPoint(
            second_time,
            request.surface.frame,
            second_position,
            handoff.velocity_m_s,
            handoff.angular_velocity_rad_s,
            GroundPhase.SKID,
        ),
    )
    segment = BounceAirSegment(
        first_time,
        second_time,
        first_position,
        second_position,
        (
            (second_position[0] - first_position[0]) ** 2
            + (second_position[2] - first_position[2]) ** 2
        )
        ** 0.5,
        True,
    )
    return RepeatedBounceResult(
        request.request_id,
        request.surface.surface_id,
        request.surface.frame,
        "tools-ground-impact-bounce",
        "1.0.0",
        points,
        events,
        impacts,
        (segment,),
        handoff,
        BounceTermination(
            BounceTerminationReason.SETTLED_TO_SKID,
            second_time,
            second_time - first_time,
        ),
        ("test prefix",),
    )


def _event_from_impact(
    sequence: int,
    event_type: GroundEventType,
    impact: ImpactImpulseResult,
) -> GroundEvent:
    """Build the exact event ledger row corresponding to an impact record."""
    before = impact.state_before
    after = impact.state_after
    return GroundEvent(
        sequence,
        event_type,
        before.time_s,
        before.frame,
        before.position_m,
        before.velocity_m_s,
        after.velocity_m_s,
        before.angular_velocity_rad_s,
        after.angular_velocity_rad_s,
    )


__all__ = [
    "_contact",
    "_failed_result",
    "_request",
    "_result",
    "_settled_prefix",
    "_surface",
    "_surface_run_request",
]
