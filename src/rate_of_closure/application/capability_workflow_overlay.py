"""Lossless editable overlay for capability workflow documents."""

from dataclasses import replace

from rate_of_closure.application.capability_workflow import (
    CapabilityWorkflowDocument,
    CapabilityWorkflowInputs,
    capability_workflow_inputs,
)
from shared.python.swing_sim.flight.capability_contract import (
    OptimizationRequest,
    PlayerCapabilityProfile,
)
from shared.python.swing_sim.flight.capability_flight_evaluator import (
    CapabilityFlightEvaluatorConfig,
)


def _identifier(value: str, name: str) -> str:
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{name} must be nonempty")
    return parsed


def _overlay_profile(
    profile: PlayerCapabilityProfile, inputs: CapabilityWorkflowInputs
) -> PlayerCapabilityProfile:
    values = {
        "ball_speed": (inputs.ball_speed_mps, inputs.ball_speed_std_mps),
        "launch_angle": (inputs.launch_angle_deg, inputs.launch_angle_std_deg),
        "launch_direction": (
            inputs.launch_direction_deg,
            inputs.launch_direction_std_deg,
        ),
    }
    source_club = profile.clubs[0]
    parameters = tuple(
        replace(
            parameter,
            baseline=values[parameter.parameter_id][0],
            standard_deviation=values[parameter.parameter_id][1],
        )
        for parameter in source_club.parameters
    )
    club_id = _identifier(inputs.club_id, "club_id")
    club = replace(source_club, club_id=club_id, parameters=parameters)
    return replace(
        profile,
        profile_id=_identifier(inputs.profile_id, "profile_id"),
        clubs=(club,),
    )


def _overlay_request(
    request: OptimizationRequest, inputs: CapabilityWorkflowInputs, club_id: str
) -> OptimizationRequest:
    target = replace(
        request.target,
        distance_m=inputs.target_distance_m,
        lateral_m=inputs.target_lateral_m,
        radius_m=inputs.target_radius_m,
    )
    return replace(
        request,
        objective=inputs.objective,
        club_ids=(club_id,),
        target=target,
        candidate_budget=inputs.candidate_budget,
        ensemble_size=inputs.ensemble_size,
        alternatives_count=inputs.alternatives_count,
        seed=inputs.seed,
    )


def _overlay_config(
    config: CapabilityFlightEvaluatorConfig,
    inputs: CapabilityWorkflowInputs,
    club_id: str,
) -> CapabilityFlightEvaluatorConfig:
    source_spin = config.spin_defaults[0]
    spin = replace(
        source_spin,
        club_id=club_id,
        total_spin_rpm=inputs.total_spin_rpm,
        spin_axis_tilt_deg=inputs.spin_axis_tilt_deg,
    )
    return replace(
        config,
        max_time_s=inputs.max_time_s,
        trajectory_sample_interval_s=inputs.trajectory_sample_interval_s,
        spin_defaults=(spin,),
    )


def overlay_capability_workflow_inputs(
    document: CapabilityWorkflowDocument,
    inputs: CapabilityWorkflowInputs,
) -> CapabilityWorkflowDocument:
    """Overlay editable controls while retaining the complete validated basis."""
    if not isinstance(document, CapabilityWorkflowDocument):
        raise TypeError("document must be a CapabilityWorkflowDocument")
    if not isinstance(inputs, CapabilityWorkflowInputs):
        raise TypeError("inputs must be CapabilityWorkflowInputs")
    capability_workflow_inputs(document)
    profile = _overlay_profile(document.profile, inputs)
    club_id = profile.clubs[0].club_id
    return CapabilityWorkflowDocument(
        profile,
        _overlay_request(document.request, inputs, club_id),
        _overlay_config(document.evaluator_config, inputs, club_id),
    )


__all__ = ["overlay_capability_workflow_inputs"]
