"""Rate-specific injected evaluator for shared bounded Morris execution."""

from __future__ import annotations

from dataclasses import dataclass

from rate_of_closure.simulation import (
    BallSupportMode,
    ContactMode,
    SimulationConfig,
    run_simulation,
)
from rate_of_closure.variation.request_builder import (
    apply_global_simulation_values,
)
from rate_of_closure.variation.simulation_types import ALL_OUTPUT_NAMES, APP_FRAME_ID
from rate_of_closure.variation.trial_projection import (
    SimulationExecutor,
    capture_simulation,
    project_simulation_outcome,
)
from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    CATEGORY_BALL_SETUP,
    CATEGORY_CLUB,
    CATEGORY_DELIVERY,
    CATEGORY_SWING,
    MorrisDesign,
    MorrisEvaluation,
    MorrisExecutionOptions,
    MorrisObservations,
    MorrisOutput,
    MorrisSample,
    evaluate_morris_design,
    variable_registry,
)

_IMPACT_TIME_OFFSET = f"{CATEGORY_SWING}.impact_time_offset_s"
_TEE_HEIGHT = f"{CATEGORY_BALL_SETUP}.tee_height_m"
RATE_MORRIS_VARIABLE_KEYS = frozenset(
    {
        f"{CATEGORY_SWING}.yaw_deg",
        f"{CATEGORY_SWING}.side_tilt_deg",
        f"{CATEGORY_SWING}.forward_tilt_deg",
        f"{CATEGORY_SWING}.damping_shoulder",
        f"{CATEGORY_SWING}.damping_wrist",
        f"{CATEGORY_DELIVERY}.impact_offset_toe_mm",
        f"{CATEGORY_DELIVERY}.impact_offset_high_mm",
        f"{CATEGORY_CLUB}.head_mass_kg",
        f"{CATEGORY_CLUB}.head_moi_kg_m2",
        _TEE_HEIGHT,
    }
)

_OUTPUT_UNITS = (
    "s",
    "m",
    "m",
    "s",
    "m/s",
    "deg",
    "deg",
    "deg",
    "mph",
    "deg",
    "deg",
    "rpm",
    "m",
    "m",
    "m",
    "s",
    "deg",
)
_APP_FRAME_OUTPUTS = frozenset(
    {
        "spin_axis_tilt_deg",
        "launch_angle_deg",
        "launch_azimuth_deg",
        "carry_m",
        "lateral_m",
        "max_height_m",
        "landing_angle_deg",
    }
)


def _output_kind(index: int) -> str:
    if index < 3:
        return "scalar"
    if index < 8:
        return "impact"
    return "shot-outcome"


RATE_MORRIS_OUTPUTS = tuple(
    MorrisOutput(
        name=name,
        unit=unit,
        target_kind=_output_kind(index),
        coordinate_frame=APP_FRAME_ID if name in _APP_FRAME_OUTPUTS else None,
    )
    for index, (name, unit) in enumerate(
        zip(ALL_OUTPUT_NAMES, _OUTPUT_UNITS, strict=True)
    )
)


@dataclass(frozen=True)
class RateMorrisEvaluator:
    """Map immutable Morris samples into fixed-ball Rate simulations."""

    design: MorrisDesign
    base_config: SimulationConfig
    executor: SimulationExecutor = run_simulation

    def __post_init__(self) -> None:
        require(isinstance(self.design, MorrisDesign), "design must be MorrisDesign")
        require(
            isinstance(self.base_config, SimulationConfig),
            "base_config must be SimulationConfig",
        )
        require(callable(self.executor), "executor must be callable")
        _validate_base_config(self.base_config)
        _validate_factors(self.design, self.base_config)

    def __call__(self, sample: MorrisSample) -> MorrisEvaluation:
        """Evaluate one physical sample with exact Rate scalar projection."""
        values = _sample_values(self.design, sample)
        config = apply_global_simulation_values(self.base_config, values)
        capture = capture_simulation(config, self.executor)
        outcome = project_simulation_outcome(sample.ordinal, capture)
        failure_message = outcome.failure_message
        if failure_message is not None:
            failure_message = " ".join(failure_message.split())[:1024].strip()
            if not failure_message:
                failure_message = outcome.failure_type
        return MorrisEvaluation(
            outcome.status.value,
            outcome.values,
            outcome.failure_type,
            failure_message,
        )


def _validate_base_config(config: SimulationConfig) -> None:
    require(
        config.source_kind == "double_pendulum",
        "Rate Morris execution requires the double_pendulum source",
        config.source_kind,
    )
    require(
        config.contact_mode is ContactMode.FIXED_BALL_CONTACT,
        "Rate Morris execution requires ContactMode.FIXED_BALL_CONTACT",
        config.contact_mode,
    )


def _validate_factors(design: MorrisDesign, config: SimulationConfig) -> None:
    factors = design.factors
    require(
        all(
            factor.source_time_window_s is None and not factor.source_point_ids
            for factor in factors
        ),
        "Rate Morris execution supports only global factors",
    )
    keys = tuple(factor.variable_key for factor in factors)
    require(len(set(keys)) == len(keys), "factors require unique variable_key values")
    require(
        _IMPACT_TIME_OFFSET not in keys,
        "fixed-ball contact ignores impact_time_offset_s",
    )
    unsupported = sorted(set(keys) - RATE_MORRIS_VARIABLE_KEYS)
    require(not unsupported, "factors are not supported by Rate Morris", unsupported)
    registry = variable_registry()
    require(
        all(factor.unit == registry[factor.variable_key].unit for factor in factors),
        "factor unit must match the registered unit",
    )
    require(
        _TEE_HEIGHT not in keys
        or config.ball_setup.support_mode is BallSupportMode.TEE,
        "tee_height_m requires Tee support",
        config.ball_setup.support_mode,
    )


def _sample_values(design: MorrisDesign, sample: MorrisSample) -> dict[str, float]:
    require(isinstance(sample, MorrisSample), "sample must be MorrisSample")
    require(
        sample.factors == design.factors,
        "sample factors must match evaluator design factors",
    )
    points_per_trajectory = len(design.factors) + 1
    expected_indices = divmod(sample.ordinal, points_per_trajectory)
    require(
        expected_indices == (sample.trajectory_index, sample.point_index)
        and expected_indices[0] < design.trajectories,
        "sample ordinal and trajectory/point identity must agree",
        expected_indices,
    )
    normalized = design.normalized_points[expected_indices]
    expected_values = {
        factor.spec_id: float(
            factor.lower + normalized[index] * (factor.upper - factor.lower)
        )
        for index, factor in enumerate(design.factors)
    }
    require(
        dict(sample.physical_values) == expected_values,
        "sample physical values must match the design point",
    )
    return {
        factor.variable_key: sample.physical_values[factor.spec_id]
        for factor in design.factors
    }


def evaluate_rate_morris_design(
    design: MorrisDesign,
    base_config: SimulationConfig,
    options: MorrisExecutionOptions | None = None,
    executor: SimulationExecutor = run_simulation,
) -> MorrisObservations:
    """Execute a validated Rate fixed-ball Morris design."""
    evaluator = RateMorrisEvaluator(design, base_config, executor)
    return evaluate_morris_design(design, RATE_MORRIS_OUTPUTS, evaluator, options)


__all__ = [
    "RATE_MORRIS_OUTPUTS",
    "RATE_MORRIS_VARIABLE_KEYS",
    "RateMorrisEvaluator",
    "evaluate_rate_morris_design",
]
