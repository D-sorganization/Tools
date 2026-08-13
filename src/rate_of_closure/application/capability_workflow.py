"""Versioned authoring and persistence facade for capability optimization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from rate_of_closure.application.capability_workflow_wire import (
    validate_capability_workflow_wire,
)
from shared.python.swing_sim.flight.capability_contract import (
    CapabilityObjective,
    CapabilityParameter,
    ClubCapability,
    OptimizationRequest,
    PlayerCapabilityProfile,
    TargetDefinition,
)
from shared.python.swing_sim.flight.capability_flight_evaluator import (
    CapabilityFlightEvaluatorConfig,
    CapabilitySpinDefault,
)

CAPABILITY_WORKFLOW_SCHEMA_VERSION = "capability-optimization-workflow/v1"
MAX_WORKFLOW_OBSERVATIONS = 100_000
_PROVENANCE = "rate-of-closure/capability-workflow/user-authored/v1"
CAPABILITY_WORKFLOW_NUMERIC_BOUNDS = {
    "ball_speed_mps": (1.0, 100.0),
    "ball_speed_std_mps": (0.0, 30.0),
    "launch_angle_deg": (-10.0, 45.0),
    "launch_angle_std_deg": (0.0, 30.0),
    "launch_direction_deg": (-30.0, 30.0),
    "launch_direction_std_deg": (0.0, 30.0),
    "total_spin_rpm": (0.0, 20_000.0),
    "spin_axis_tilt_deg": (-90.0, 90.0),
    "target_distance_m": (0.1, 1_000.0),
    "target_lateral_m": (-500.0, 500.0),
    "target_radius_m": (0.1, 500.0),
    "max_time_s": (0.001, 120.0),
    "trajectory_sample_interval_s": (0.001, 0.1),
}
CAPABILITY_WORKFLOW_INTEGER_BOUNDS = {
    "candidate_budget": (1, MAX_WORKFLOW_OBSERVATIONS),
    "ensemble_size": (1, MAX_WORKFLOW_OBSERVATIONS),
    "alternatives_count": (1, MAX_WORKFLOW_OBSERVATIONS),
    "seed": (0, 2**31 - 1),
}


@dataclass(frozen=True)
class CapabilityWorkflowInputs:
    """Editable single-club workflow inputs expressed in canonical units."""

    profile_id: str = "representative-driver-profile"
    club_id: str = "driver"
    ball_speed_mps: float = 67.0
    ball_speed_std_mps: float = 1.5
    launch_angle_deg: float = 12.5
    launch_angle_std_deg: float = 1.0
    launch_direction_deg: float = 0.0
    launch_direction_std_deg: float = 1.5
    total_spin_rpm: float = 2600.0
    spin_axis_tilt_deg: float = 0.0
    target_distance_m: float = 230.0
    target_lateral_m: float = 0.0
    target_radius_m: float = 12.0
    objective: CapabilityObjective = CapabilityObjective.MAXIMIZE_TARGET_HOLD
    candidate_budget: int = 8
    ensemble_size: int = 12
    alternatives_count: int = 3
    seed: int = 4197
    max_time_s: float = 10.0
    trajectory_sample_interval_s: float = 0.01


@dataclass(frozen=True)
class CapabilityWorkflowDocument:
    """Strict persisted profile, request, and evaluator configuration."""

    profile: PlayerCapabilityProfile
    request: OptimizationRequest
    evaluator_config: CapabilityFlightEvaluatorConfig
    schema_version: str = CAPABILITY_WORKFLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_WORKFLOW_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        requested = set(self.request.club_ids)
        available = {club.club_id for club in self.profile.clubs}
        if not requested <= available:
            raise ValueError("request club_ids must exist in the profile")
        spin_clubs = {item.club_id for item in self.evaluator_config.spin_defaults}
        if spin_clubs != requested:
            raise ValueError(
                "spin default club_ids must exactly match request club_ids"
            )
        attempts = self.request.candidate_budget * self.request.ensemble_size
        if attempts > MAX_WORKFLOW_OBSERVATIONS:
            raise ValueError(
                f"workflow may not exceed {MAX_WORKFLOW_OBSERVATIONS} observations"
            )
        _validate_authoring_inputs(capability_workflow_inputs(self))

    def to_wire(self) -> dict[str, object]:
        """Return the exact versioned persistence representation."""
        return {
            "evaluator_config": _config_to_wire(self.evaluator_config),
            "profile": self.profile.to_dict(),
            "request": self.request.to_dict(),
            "schema_version": self.schema_version,
        }


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be nonempty text")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{name} must be nonempty")
    return parsed


def _validate_authoring_inputs(inputs: CapabilityWorkflowInputs) -> None:
    for key, (minimum, maximum) in CAPABILITY_WORKFLOW_NUMERIC_BOUNDS.items():
        value = _finite(getattr(inputs, key), key)
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must lie within [{minimum:g}, {maximum:g}]")
    for key, (minimum, maximum) in CAPABILITY_WORKFLOW_INTEGER_BOUNDS.items():
        value = getattr(inputs, key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must lie within [{minimum}, {maximum}]")


def _parameter(
    parameter_id: str, unit: str, baseline: float, standard_deviation: float
) -> CapabilityParameter:
    domains = {
        "ball_speed": (1.0, 100.0, 20.0, 90.0),
        "launch_angle": (-10.0, 45.0, 0.0, 35.0),
        "launch_direction": (-30.0, 30.0, -15.0, 15.0),
    }
    lower, upper, evidence_lower, evidence_upper = domains[parameter_id]
    return CapabilityParameter(
        parameter_id,
        unit,
        lower,
        upper,
        evidence_lower,
        evidence_upper,
        _finite(baseline, f"{parameter_id} baseline"),
        0.0,
        _finite(standard_deviation, f"{parameter_id} standard deviation"),
    )


def _profile(inputs: CapabilityWorkflowInputs) -> PlayerCapabilityProfile:
    if inputs.ball_speed_mps <= 0.0:
        raise ValueError("ball_speed_mps must be greater than zero")
    parameters = (
        _parameter(
            "ball_speed", "m/s", inputs.ball_speed_mps, inputs.ball_speed_std_mps
        ),
        _parameter(
            "launch_angle",
            "deg",
            inputs.launch_angle_deg,
            inputs.launch_angle_std_deg,
        ),
        _parameter(
            "launch_direction",
            "deg",
            inputs.launch_direction_deg,
            inputs.launch_direction_std_deg,
        ),
    )
    club_id = _text(inputs.club_id, "club_id")
    club = ClubCapability(
        club_id,
        parameters,
        "correlation",
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        _PROVENANCE,
        0.8,
    )
    return PlayerCapabilityProfile(
        _text(inputs.profile_id, "profile_id"), (club,), _PROVENANCE, 0.8
    )


def _request(inputs: CapabilityWorkflowInputs, club_id: str) -> OptimizationRequest:
    target = TargetDefinition(
        "green",
        _finite(inputs.target_distance_m, "target_distance_m"),
        _finite(inputs.target_lateral_m, "target_lateral_m"),
        _finite(inputs.target_radius_m, "target_radius_m"),
        _finite(inputs.target_radius_m, "target_radius_m"),
        _finite(inputs.target_radius_m, "target_radius_m"),
    )
    return OptimizationRequest(
        f"capability-{_text(inputs.profile_id, 'profile_id')}",
        CapabilityObjective(inputs.objective),
        (club_id,),
        target,
        inputs.candidate_budget,
        inputs.ensemble_size,
        inputs.alternatives_count,
        inputs.seed,
        0.9,
        0.8,
    )


def _config(
    inputs: CapabilityWorkflowInputs, club_id: str
) -> CapabilityFlightEvaluatorConfig:
    if not -90.0 <= inputs.spin_axis_tilt_deg <= 90.0:
        raise ValueError("spin_axis_tilt_deg must lie within [-90, 90]")
    spin = CapabilitySpinDefault(
        club_id,
        _finite(inputs.total_spin_rpm, "total_spin_rpm"),
        _finite(inputs.spin_axis_tilt_deg, "spin_axis_tilt_deg"),
        _PROVENANCE,
    )
    return CapabilityFlightEvaluatorConfig(
        _finite(inputs.max_time_s, "max_time_s"),
        _finite(inputs.trajectory_sample_interval_s, "trajectory_sample_interval_s"),
        (spin,),
    )


def build_capability_workflow(
    inputs: CapabilityWorkflowInputs,
) -> CapabilityWorkflowDocument:
    """Build a validated model-ready document from editable canonical inputs."""
    if not isinstance(inputs, CapabilityWorkflowInputs):
        raise TypeError("inputs must be CapabilityWorkflowInputs")
    profile = _profile(inputs)
    club_id = profile.clubs[0].club_id
    return CapabilityWorkflowDocument(
        profile, _request(inputs, club_id), _config(inputs, club_id)
    )


def _config_to_wire(config: CapabilityFlightEvaluatorConfig) -> dict[str, object]:
    return {
        "max_time_s": config.max_time_s,
        "spin_defaults": [
            {
                "club_id": item.club_id,
                "provenance": item.provenance,
                "spin_axis_tilt_deg": item.spin_axis_tilt_deg,
                "total_spin_rpm": item.total_spin_rpm,
            }
            for item in config.spin_defaults
        ],
        "trajectory_sample_interval_s": config.trajectory_sample_interval_s,
    }


def _config_from_wire(payload: dict[str, Any]) -> CapabilityFlightEvaluatorConfig:
    expected = {"max_time_s", "spin_defaults", "trajectory_sample_interval_s"}
    if set(payload) != expected or not isinstance(payload["spin_defaults"], list):
        raise ValueError("evaluator_config fields do not match v1 schema")
    defaults = []
    for source in payload["spin_defaults"]:
        if not isinstance(source, dict) or set(source) != {
            "club_id",
            "provenance",
            "spin_axis_tilt_deg",
            "total_spin_rpm",
        }:
            raise ValueError("spin default fields do not match v1 schema")
        defaults.append(
            CapabilitySpinDefault(
                _text(source["club_id"], "spin default club_id"),
                _finite(source["total_spin_rpm"], "total_spin_rpm"),
                _finite(source["spin_axis_tilt_deg"], "spin_axis_tilt_deg"),
                _text(source["provenance"], "spin default provenance"),
            )
        )
    return CapabilityFlightEvaluatorConfig(
        _finite(payload["max_time_s"], "max_time_s"),
        _finite(
            payload["trajectory_sample_interval_s"],
            "trajectory_sample_interval_s",
        ),
        tuple(defaults),
    )


def capability_workflow_json(document: CapabilityWorkflowDocument) -> str:
    """Serialize a workflow deterministically for local persistence."""
    if not isinstance(document, CapabilityWorkflowDocument):
        raise TypeError("document must be CapabilityWorkflowDocument")
    return json.dumps(document.to_wire(), sort_keys=True, separators=(",", ":"))


def capability_workflow_from_json(source: str) -> CapabilityWorkflowDocument:
    """Parse one exact workflow document and reject unknown fields."""
    payload = validate_capability_workflow_wire(json.loads(source))
    return CapabilityWorkflowDocument(
        PlayerCapabilityProfile.from_dict(payload["profile"]),
        OptimizationRequest.from_dict(payload["request"]),
        _config_from_wire(payload["evaluator_config"]),
        _text(payload["schema_version"], "schema_version"),
    )


def capability_workflow_inputs(
    document: CapabilityWorkflowDocument,
) -> CapabilityWorkflowInputs:
    """Project an editable single-club document back into form inputs."""
    if (
        len(document.profile.clubs) != 1
        or len(document.evaluator_config.spin_defaults) != 1
    ):
        raise ValueError(
            "interactive workflow supports exactly one club and spin default"
        )
    club = document.profile.clubs[0]
    parameters = {item.parameter_id: item for item in club.parameters}
    if set(parameters) != {"ball_speed", "launch_angle", "launch_direction"}:
        raise ValueError("interactive workflow requires the three launch parameters")
    spin = document.evaluator_config.spin_defaults[0]
    target = document.request.target
    return CapabilityWorkflowInputs(
        profile_id=document.profile.profile_id,
        club_id=club.club_id,
        ball_speed_mps=parameters["ball_speed"].baseline,
        ball_speed_std_mps=parameters["ball_speed"].standard_deviation,
        launch_angle_deg=parameters["launch_angle"].baseline,
        launch_angle_std_deg=parameters["launch_angle"].standard_deviation,
        launch_direction_deg=parameters["launch_direction"].baseline,
        launch_direction_std_deg=parameters["launch_direction"].standard_deviation,
        total_spin_rpm=spin.total_spin_rpm,
        spin_axis_tilt_deg=spin.spin_axis_tilt_deg,
        target_distance_m=target.distance_m,
        target_lateral_m=target.lateral_m,
        target_radius_m=target.radius_m,
        objective=document.request.objective,
        candidate_budget=document.request.candidate_budget,
        ensemble_size=document.request.ensemble_size,
        alternatives_count=document.request.alternatives_count,
        seed=document.request.seed,
        max_time_s=document.evaluator_config.max_time_s,
        trajectory_sample_interval_s=document.evaluator_config.trajectory_sample_interval_s,
    )


__all__ = [
    "CAPABILITY_WORKFLOW_SCHEMA_VERSION",
    "CAPABILITY_WORKFLOW_INTEGER_BOUNDS",
    "CAPABILITY_WORKFLOW_NUMERIC_BOUNDS",
    "CapabilityWorkflowDocument",
    "CapabilityWorkflowInputs",
    "build_capability_workflow",
    "capability_workflow_from_json",
    "capability_workflow_inputs",
    "capability_workflow_json",
]
