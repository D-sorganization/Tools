"""Deterministic centered-club delivery to impact and flight evaluator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from shared.python.swing_sim.impact import (
    DeliveryParameters,
    ImpactModelType,
    ImpactSolverAPI,
    derive_delivery,
)

from .frames import to_flight_frame
from .impact_solution_contract import (
    IMPACT_MODEL_ID,
    ClubProfileId,
    ForwardEvaluation,
    ForwardStatus,
    ImpactSolutionRequest,
    ModelAvailability,
    ModelManifest,
)
from .inverse_contract import EvaluatedMetric
from .launch import derive_launch_conditions
from .pipeline import simulate
from .registry import FlightModelType
from .result_contract import FlightMetricId
from .spin_axis_convention import spin_axis_tilt_deg

_APPROACH_EPSILON_MPS = 1e-9
_PIPELINE_PROVENANCE = "Tools delivery->rigid-body-impact->flight pipeline"


@dataclass(frozen=True)
class _ClubSpec:
    defaults: Mapping[str, float]
    mass_kg: float
    moi_kg_m2: float
    provenance: str


_CLUB_SPECS = MappingProxyType(
    {
        ClubProfileId.CENTERED_DRIVER: _ClubSpec(
            MappingProxyType(
                {
                    "attack_angle_deg": 2.0,
                    "club_path_deg": 0.0,
                    "clubhead_speed_mps": 44.0,
                    "dynamic_loft_deg": 12.0,
                    "face_angle_deg": 0.0,
                }
            ),
            0.200,
            4.5e-4,
            "representative centered driver; user-fit equipment pending",
        ),
        ClubProfileId.CENTERED_IRON: _ClubSpec(
            MappingProxyType(
                {
                    "attack_angle_deg": -4.0,
                    "club_path_deg": 0.0,
                    "clubhead_speed_mps": 36.0,
                    "dynamic_loft_deg": 25.0,
                    "face_angle_deg": 0.0,
                }
            ),
            0.255,
            3.0e-4,
            "representative centered iron; user-fit equipment pending",
        ),
    }
)


def _metric(metric_id: FlightMetricId, value: float, stage: str) -> EvaluatedMetric:
    return EvaluatedMetric(metric_id, value, f"{_PIPELINE_PROVENANCE}:{stage}")


def _spin_axis_tilt_deg(spin_app: np.ndarray) -> float:
    tilt = spin_axis_tilt_deg(spin_app)
    return 0.0 if tilt is None else tilt


class CenteredClubDeliveryAdapter:
    """Evaluate only declared centered driver or iron delivery variables.

    The adapter intentionally does not expose shaft, contact-offset, turf or
    flexible-head variables. Those require independently validated models.
    """

    def __init__(self, request: ImpactSolutionRequest) -> None:
        self._request = request
        self._spec = _CLUB_SPECS[request.club_profile_id]
        available = request.flight_model_id in {item.value for item in FlightModelType}
        self._manifest = ModelManifest(
            IMPACT_MODEL_ID,
            ModelAvailability.AVAILABLE,
            request.flight_model_id,
            (
                ModelAvailability.AVAILABLE
                if available
                else ModelAvailability.UNAVAILABLE
            ),
            (
                ("club_profile", self._spec.provenance),
                ("frame_adapter", "swing_sim.flight.frames.to_flight_frame"),
                ("pipeline", _PIPELINE_PROVENANCE),
            ),
        )

    @property
    def model_manifest(self) -> ModelManifest:
        """Return immutable model identity and availability metadata."""
        return self._manifest

    def _parameters(self, supplied: Mapping[str, float]) -> DeliveryParameters:
        expected = {
            item.parameter_id for item in self._request.inverse_request.variables
        }
        if set(supplied) != expected:
            raise ValueError(
                "forward parameters must match declared decision variables"
            )
        merged = dict(self._spec.defaults)
        variables = {
            item.parameter_id: item for item in self._request.inverse_request.variables
        }
        for parameter_id, value in supplied.items():
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{parameter_id} must be finite")
            variable = variables[parameter_id]
            if not variable.lower_bound <= value <= variable.upper_bound:
                raise ValueError(f"{parameter_id} must remain within declared bounds")
            merged[parameter_id] = float(value)
        return DeliveryParameters(
            clubhead_speed_mps=merged["clubhead_speed_mps"],
            club_path_deg=merged["club_path_deg"],
            face_angle_deg=merged["face_angle_deg"],
            attack_angle_deg=merged["attack_angle_deg"],
            dynamic_loft_deg=merged["dynamic_loft_deg"],
        )

    def _incomplete(self, status: ForwardStatus, reason: str) -> ForwardEvaluation:
        return ForwardEvaluation(status, (), (), self._manifest, reason)

    def evaluate(self, supplied: Mapping[str, float]) -> ForwardEvaluation:
        """Run one centered delivery through impact, launch and flight.

        Args:
            supplied: Exact declared decision-variable mapping.

        Returns:
            Complete scalar metrics or a typed no-impact/model diagnostic.
        """
        if self._manifest.flight_status is ModelAvailability.UNAVAILABLE:
            return self._incomplete(
                ForwardStatus.MODEL_UNAVAILABLE, "unknown_flight_model"
            )
        try:
            parameters = self._parameters(supplied)
            delivery = derive_delivery(parameters)
            approach = float(np.dot(delivery.clubhead_velocity, delivery.face_normal))
            if approach <= _APPROACH_EPSILON_MPS:
                return self._incomplete(
                    ForwardStatus.NO_IMPACT, "nonpositive_normal_approach_speed"
                )
            solver = ImpactSolverAPI(ImpactModelType.RIGID_BODY)
            post = solver.solve_impact(
                timestamp=self._request.impact_event_time_s,
                clubhead_velocity=delivery.clubhead_velocity,
                clubhead_orientation=delivery.face_normal,
                clubhead_mass=self._spec.mass_kg,
                clubhead_moi=self._spec.moi_kg_m2,
                record=False,
            )
            launch = derive_launch_conditions(
                to_flight_frame(post.ball_velocity),
                to_flight_frame(post.ball_angular_velocity),
            )
            flight = simulate(launch, model_name=self._request.flight_model_id)
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return self._incomplete(ForwardStatus.FAILED, "forward_pipeline_error")

        launch_metrics = (
            _metric(FlightMetricId.BALL_SPEED, launch.ball_speed, "launch"),
            _metric(
                FlightMetricId.VERTICAL_LAUNCH_ANGLE,
                math.degrees(launch.launch_angle),
                "launch",
            ),
            _metric(
                FlightMetricId.LAUNCH_DIRECTION,
                -math.degrees(launch.azimuth_angle),
                "launch",
            ),
            _metric(FlightMetricId.TOTAL_SPIN, launch.spin_rate, "launch"),
            _metric(
                FlightMetricId.SPIN_AXIS_TILT,
                _spin_axis_tilt_deg(post.ball_angular_velocity),
                "launch",
            ),
        )
        flight_metrics = (
            _metric(FlightMetricId.CARRY_DISTANCE, flight.carry_distance, "flight"),
            _metric(FlightMetricId.CARRY_OFFLINE, -flight.lateral_deviation, "flight"),
            _metric(FlightMetricId.APEX_HEIGHT, flight.max_height, "flight"),
            _metric(FlightMetricId.FLIGHT_TIME, flight.flight_time, "flight"),
            _metric(FlightMetricId.LANDING_ANGLE, flight.landing_angle, "flight"),
        )
        return ForwardEvaluation(
            ForwardStatus.COMPLETE,
            launch_metrics,
            flight_metrics,
            self._manifest,
        )


__all__ = ["CenteredClubDeliveryAdapter"]
