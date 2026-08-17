"""Authoritative flight, bounce, and regional-ground composition."""

from __future__ import annotations

from dataclasses import dataclass

from shared.python.swing_sim.ground.bounce_request_wire import (
    RepeatedBounceRequestResultPair,
)
from shared.python.swing_sim.ground.bounce_types import (
    BounceModelSettings,
    BounceTerminationReason,
)
from shared.python.swing_sim.ground.contract_records import GroundSimulationResult
from shared.python.swing_sim.ground.regional_execution import (
    RegionalGroundExecutionOptions,
    execute_regional_ground,
)
from shared.python.swing_sim.ground.regional_execution_records import (
    RegionalGroundExecutionResult,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    GroundRegionalMaterialPlanRequest,
    regional_plan_request_sha256,
)

from .ground_bounce_execution import execute_repeated_bounce_from_flight
from .ground_transfer import (
    FlightGroundTransferSettings,
    launch_relative_surface,
)
from .types import FlightResult, LaunchConditions

FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION = "flight-regional-ground-pipeline/v1"


@dataclass(frozen=True)
class FlightRegionalGroundPipelineResult:
    """Strict bounded composition evidence without a duplicate wire schema."""

    bounce_result: RepeatedBounceRequestResultPair
    regional_plan: GroundRegionalMaterialPlanRequest
    ground_request_sha256: str
    repeated_bounce_execution_input_sha256: str
    regional_plan_sha256: str
    regional_result: RegionalGroundExecutionResult | None
    contract_version: str = FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        """Bind nested phase evidence and forbid fabricated phase execution."""
        if type(self.bounce_result) is not RepeatedBounceRequestResultPair:
            raise ValueError(
                "bounce_result must be an exact RepeatedBounceRequestResultPair"
            )
        if type(self.regional_plan) is not GroundRegionalMaterialPlanRequest:
            raise ValueError(
                "regional_plan must be an exact GroundRegionalMaterialPlanRequest"
            )
        if self.contract_version != FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION:
            raise ValueError("unsupported contract_version")
        request = self.bounce_result.request
        if self.ground_request_sha256 != request.ground_request_sha256:
            raise ValueError("ground_request_sha256 must match the bounce request")
        if (
            self.repeated_bounce_execution_input_sha256
            != request.execution_input_sha256
        ):
            raise ValueError(
                "repeated_bounce_execution_input_sha256 must match the bounce request"
            )
        expected_plan_digest = regional_plan_request_sha256(self.regional_plan)
        if self.regional_plan_sha256 != expected_plan_digest:
            raise ValueError("regional_plan_sha256 must match the regional plan")
        if self.regional_plan.base_surface != request.ground_request.surface:
            raise ValueError("regional plan base surface must match the ground request")
        self._validate_phase_outcome()

    def _validate_phase_outcome(self) -> None:
        reason = self.bounce_result.result.termination.reason
        if reason is not BounceTerminationReason.SETTLED_TO_SKID:
            if self.regional_result is not None:
                raise ValueError("non-settled bounce forbids regional result")
            return
        if self.regional_result is None:
            raise ValueError("settled bounce requires regional result")
        if type(self.regional_result) is not RegionalGroundExecutionResult:
            raise ValueError(
                "regional_result must be an exact RegionalGroundExecutionResult"
            )
        regional = self.regional_result
        request = self.bounce_result.request
        if regional.ground_request_sha256 != self.ground_request_sha256:
            raise ValueError("regional result must match the ground request digest")
        if (
            regional.regional_plan != self.regional_plan
            or regional.regional_plan_sha256 != self.regional_plan_sha256
        ):
            raise ValueError("regional result must match the regional plan")
        if (regional.request_id, regional.surface_id) != (
            request.request_id,
            request.surface_id,
        ):
            raise ValueError("regional result must match request identities")
        if not regional.model_id.startswith(f"{request.model_id}+"):
            raise ValueError("regional result must include the bounce model identity")
        if not regional.model_version.startswith(f"{request.model_version}+"):
            raise ValueError("regional result must include the bounce model version")

    @property
    def ground_result(self) -> GroundSimulationResult | None:
        """Return the qualified ground result when regional execution produced one."""
        if self.regional_result is None:
            return None
        return self.regional_result.ground_result


def _validate_inputs(
    flight: FlightResult,
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    plan: GroundRegionalMaterialPlanRequest,
    options: RegionalGroundExecutionOptions,
    capture_speed_m_s: float,
) -> float:
    if type(flight) is not FlightResult:
        raise ValueError("flight must be an exact FlightResult")
    if type(launch) is not LaunchConditions:
        raise ValueError("launch must be an exact LaunchConditions")
    if type(transfer) is not FlightGroundTransferSettings:
        raise ValueError("transfer must be an exact FlightGroundTransferSettings")
    if type(plan) is not GroundRegionalMaterialPlanRequest:
        raise ValueError("plan must be an exact GroundRegionalMaterialPlanRequest")
    if type(options) is not RegionalGroundExecutionOptions:
        raise ValueError("options must be an exact RegionalGroundExecutionOptions")
    # `float(...)` is for the type checker, not the value: CI runs mypy with
    # `--follow-imports=skip`, so `BounceModelSettings` resolves to `Any` and
    # its attribute would be returned as `Any` from a `-> float` function.
    # The settings object has already validated the number.
    validated_capture = float(
        BounceModelSettings(capture_speed_m_s=capture_speed_m_s).capture_speed_m_s
    )
    expected_surface = launch_relative_surface(
        transfer.surface,
        launch.ball_radius,
        launch.ball_setup,
    )
    if plan.base_surface != expected_surface:
        raise ValueError(
            "plan.base_surface must equal the launch-relative transfer surface"
        )
    return validated_capture


def execute_regional_ground_from_flight(
    flight: FlightResult,
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    plan: GroundRegionalMaterialPlanRequest,
    capture_speed_m_s: float = 0.05,
    *,
    options: RegionalGroundExecutionOptions | None = None,
) -> FlightRegionalGroundPipelineResult:
    """Run existing phase authorities in order without duplicating physics."""
    selected = RegionalGroundExecutionOptions() if options is None else options
    validated_capture = _validate_inputs(
        flight,
        launch,
        transfer,
        plan,
        selected,
        capture_speed_m_s,
    )
    plan_digest = regional_plan_request_sha256(plan)
    bounce = execute_repeated_bounce_from_flight(
        flight,
        launch,
        transfer,
        validated_capture,
        is_cancelled=selected.is_cancelled,
    )
    regional = None
    if bounce.result.termination.reason is BounceTerminationReason.SETTLED_TO_SKID:
        regional = execute_regional_ground(
            bounce.request.ground_request,
            bounce.result,
            plan,
            selected,
        )
    return FlightRegionalGroundPipelineResult(
        bounce,
        plan,
        bounce.request.ground_request_sha256,
        bounce.execution_input_sha256,
        plan_digest,
        regional,
    )


__all__ = [
    "FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION",
    "FlightRegionalGroundPipelineResult",
    "execute_regional_ground_from_flight",
]
