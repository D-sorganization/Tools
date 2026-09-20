"""Open-data validation package and benchmark carry references (#4252).

Provides pinned published benchmark references across 5 standard launch
conditions (Tour Driver, Amateur Driver, 5-Iron, Wedge, High-Speed Driver),
tracks stratified model residuals, evaluates internal runtime parity
(Rust kernel vs Python canonical Waterloo/Penner), and documents physics
limitations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ._rust_facade import is_rust_available, simulate_trajectory_rust
from .registry import CANONICAL_FLIGHT_MODEL, FlightModelRegistry
from .types import LaunchConditions

VALIDATION_SCHEMA_VERSION = "flight-validation-manifest/v1"


@dataclass(frozen=True)
class BenchmarkLaunchSpec:
    """Standard physical launch specification for benchmarking."""

    condition_id: str
    label: str
    ball_speed_mps: float
    launch_angle_deg: float
    spin_rate_rpm: float
    reference_carry_m: float
    source_citation: str
    license_status: str

    def to_launch_conditions(self) -> LaunchConditions:
        """Construct typed LaunchConditions."""
        return LaunchConditions(
            ball_speed=self.ball_speed_mps,
            launch_angle=math.radians(self.launch_angle_deg),
            spin_rate=self.spin_rate_rpm,
        )


BENCHMARK_LAUNCHES: tuple[BenchmarkLaunchSpec, ...] = (
    BenchmarkLaunchSpec(
        condition_id="tour_driver",
        label="Tour Driver (PGA Tour baseline)",
        ball_speed_mps=74.0,
        launch_angle_deg=12.0,
        spin_rate_rpm=2600.0,
        reference_carry_m=248.5,
        source_citation="TrackMan Tour Averages (2022/2024); Penner (2003)",
        license_status="Open published domain benchmark data",
    ),
    BenchmarkLaunchSpec(
        condition_id="amateur_driver",
        label="Amateur Driver (Moderate speed)",
        ball_speed_mps=60.0,
        launch_angle_deg=14.0,
        spin_rate_rpm=2800.0,
        reference_carry_m=189.5,
        source_citation="TrackMan Amateur Benchmarks; USGA Technical Reports",
        license_status="Open published domain benchmark data",
    ),
    BenchmarkLaunchSpec(
        condition_id="mid_iron_5iron",
        label="5-Iron (Mid iron)",
        ball_speed_mps=55.0,
        launch_angle_deg=16.0,
        spin_rate_rpm=5000.0,
        reference_carry_m=168.0,
        source_citation="TrackMan Tour Averages; Penner (2003)",
        license_status="Open published domain benchmark data",
    ),
    BenchmarkLaunchSpec(
        condition_id="wedge",
        label="Pitching Wedge (High spin)",
        ball_speed_mps=40.0,
        launch_angle_deg=28.0,
        spin_rate_rpm=9000.0,
        reference_carry_m=112.5,
        source_citation="TrackMan Tour Averages; Penner (2003)",
        license_status="Open published domain benchmark data",
    ),
    BenchmarkLaunchSpec(
        condition_id="high_speed_driver",
        label="High-Speed Driver (Long drive / elite)",
        ball_speed_mps=80.0,
        launch_angle_deg=11.0,
        spin_rate_rpm=2400.0,
        reference_carry_m=280.0,
        source_citation="TrackMan Tour Elite Benchmarks",
        license_status="Open published domain benchmark data",
    ),
)


@dataclass(frozen=True)
class ConditionModelEvaluation:
    """Model evaluation metrics for a single launch condition."""

    carry_m: float
    max_height_m: float
    flight_time_s: float
    residual_m: float
    relative_error_pct: float


@dataclass(frozen=True)
class ModelAggregateMetrics:
    """Stratified accuracy metrics across all benchmark conditions."""

    mae_m: float
    rmse_m: float
    max_abs_residual_m: float
    max_relative_error_pct: float


@dataclass(frozen=True)
class RuntimeParityCheck:
    """Verification of parity between Rust kernel and Python canonical baseline."""

    condition_id: str
    python_carry_m: float
    rust_carry_m: float
    absolute_diff_m: float
    relative_diff_pct: float
    passed: bool


@dataclass(frozen=True)
class FlightValidationReport:
    """Comprehensive validation and benchmarking report."""

    schema_version: str
    canonical_model: str
    model_evaluations: dict[str, dict[str, ConditionModelEvaluation]]
    aggregate_metrics: dict[str, ModelAggregateMetrics]
    runtime_parity_checks: list[RuntimeParityCheck]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


def evaluate_flight_benchmarks() -> FlightValidationReport:
    """Run benchmark evaluation across all registered models and Rust fast path.

    Returns:
        FlightValidationReport detailing condition residuals, aggregate errors,
        and internal runtime parity checks.
    """
    all_models = FlightModelRegistry.get_all_models()
    model_evals: dict[str, dict[str, ConditionModelEvaluation]] = {
        model.name: {} for model in all_models
    }
    rust_available = is_rust_available()
    if rust_available:
        model_evals["tools-core Rust RK4"] = {}

    parity_checks: list[RuntimeParityCheck] = []
    canonical_model = FlightModelRegistry.get_model(CANONICAL_FLIGHT_MODEL)

    for spec in BENCHMARK_LAUNCHES:
        launch = spec.to_launch_conditions()
        ref_carry = spec.reference_carry_m

        for model in all_models:
            result = model.simulate(launch)
            carry = result.carry_distance
            residual = carry - ref_carry
            rel_err = abs(residual) / ref_carry * 100.0
            model_evals[model.name][spec.condition_id] = ConditionModelEvaluation(
                carry_m=round(carry, 3),
                max_height_m=round(result.max_height, 3),
                flight_time_s=round(result.flight_time, 3),
                residual_m=round(residual, 3),
                relative_error_pct=round(rel_err, 3),
            )

        if rust_available:
            rust_res = simulate_trajectory_rust(launch)
            r_carry = rust_res.carry_distance
            r_residual = r_carry - ref_carry
            r_rel_err = abs(r_residual) / ref_carry * 100.0
            model_evals["tools-core Rust RK4"][spec.condition_id] = (
                ConditionModelEvaluation(
                    carry_m=round(r_carry, 3),
                    max_height_m=round(rust_res.max_height, 3),
                    flight_time_s=round(rust_res.flight_time, 3),
                    residual_m=round(r_residual, 3),
                    relative_error_pct=round(r_rel_err, 3),
                )
            )

            py_carry = model_evals[canonical_model.name][spec.condition_id].carry_m
            abs_diff = abs(r_carry - py_carry)
            rel_diff = (abs_diff / py_carry) * 100.0 if py_carry > 0 else 0.0
            parity_checks.append(
                RuntimeParityCheck(
                    condition_id=spec.condition_id,
                    python_carry_m=py_carry,
                    rust_carry_m=round(r_carry, 3),
                    absolute_diff_m=round(abs_diff, 3),
                    relative_diff_pct=round(rel_diff, 3),
                    passed=bool(rel_diff <= 1.0),
                )
            )

    aggregates: dict[str, ModelAggregateMetrics] = {}
    for model_name, evals in model_evals.items():
        residuals = [e.residual_m for e in evals.values()]
        rel_errors = [e.relative_error_pct for e in evals.values()]
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        max_abs = float(np.max(np.abs(residuals)))
        max_rel = float(np.max(rel_errors))
        aggregates[model_name] = ModelAggregateMetrics(
            mae_m=round(mae, 3),
            rmse_m=round(rmse, 3),
            max_abs_residual_m=round(max_abs, 3),
            max_relative_error_pct=round(max_rel, 3),
        )

    limitations = [
        (
            "Terminal ground contact: ODE integration stops when the sphere "
            "center reaches terrain height + tee offset (z=0). Roll, bounce, "
            "and turf interaction require a dedicated ground-transfer model."
        ),
        (
            "Wind capabilities: Steady meteorological wind is supported across "
            "both the Rust kernel fast path and Python models. Non-steady wind "
            "(vertical shear gradients, temporal gusts, turbulence) is supported "
            "in Python ODE integration and raises cleanly in the Rust facade."
        ),
        (
            "Lift saturation ceiling: Golf ball lift coefficient is physically "
            "bounded by MAX_GOLF_BALL_LIFT_COEFFICIENT = 0.155 per Penner (2003) "
            "to prevent aerodynamic ballooning at extreme spin rates (>9000 RPM)."
        ),
        (
            "Spin decay: Decay rates range from 0.02 to 0.05 s^-1 across literature "
            "specifications; the canonical Waterloo/Penner baseline uses 0.05 s^-1 "
            "(MacDonald & Hanzely 1991, Penner 2003)."
        ),
    ]

    return FlightValidationReport(
        schema_version=VALIDATION_SCHEMA_VERSION,
        canonical_model=canonical_model.name,
        model_evaluations=model_evals,
        aggregate_metrics=aggregates,
        runtime_parity_checks=parity_checks,
        limitations=limitations,
    )


__all__ = [
    "BENCHMARK_LAUNCHES",
    "BenchmarkLaunchSpec",
    "ConditionModelEvaluation",
    "FlightValidationReport",
    "ModelAggregateMetrics",
    "RuntimeParityCheck",
    "VALIDATION_SCHEMA_VERSION",
    "evaluate_flight_benchmarks",
]
