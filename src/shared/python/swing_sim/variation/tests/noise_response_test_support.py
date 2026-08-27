"""Shared builders for noise-response contract and falsification tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.python.swing_sim.variation.engine import VariationDataset
from shared.python.swing_sim.variation.ensemble_types import EnsemblePositionTraces
from shared.python.swing_sim.variation.execution_metadata import make_execution_metadata
from shared.python.swing_sim.variation.noise_response import ResponseFieldInput
from shared.python.swing_sim.variation.registry import CATEGORY_LAUNCH
from shared.python.swing_sim.variation.spec import NoiseSpec, VariationPlan
from shared.python.swing_sim.variation.trace_resampling import resample_position_traces

BALL_SPEED = f"{CATEGORY_LAUNCH}.ball_speed_mph"
LAUNCH_ANGLE = f"{CATEGORY_LAUNCH}.launch_angle_deg"
INPUT_KEYS = (BALL_SPEED, LAUNCH_ANGLE)
POINT_IDS = ("swing.wrist", "swing.clubhead.reference")
TRIAL_IDS = tuple(f"trial-{index:03d}" for index in range(4))
SOURCE_SHA256 = "a" * 64


@dataclass(frozen=True)
class ResponseFixtureConfig:
    """Compact configuration for one complete two-input paired study."""

    deltas: np.ndarray
    coefficients: np.ndarray
    baseline_positions_m: np.ndarray
    baseline_valid: np.ndarray
    perturbed_valid: np.ndarray
    specs: tuple[NoiseSpec, ...] | None = None
    groups: tuple[object, ...] = ()
    input_kinds: tuple[str, ...] = ("continuous", "continuous")
    coordinate_frame: str = "swing.world"


def default_fixture_config() -> ResponseFixtureConfig:
    """Return an affine two-input, two-time, two-point golden configuration."""
    deltas = np.array(
        [
            [-2.0, -1.0, 1.0, 2.0],
            [-4.0, -2.0, 2.0, 4.0],
        ]
    )
    coefficients = np.array(
        [
            [
                [[0.10, -0.20, 0.30], [0.40, 0.00, -0.10]],
                [[0.20, -0.10, 0.00], [0.50, 0.10, -0.20]],
            ],
            [
                [[-0.30, 0.20, 0.10], [0.00, 0.25, 0.35]],
                [[-0.10, 0.30, 0.20], [0.15, 0.45, 0.05]],
            ],
        ]
    )
    baseline = np.zeros((4, 2, 2, 3), dtype=float)
    valid = np.ones((2, 4, 2), dtype=bool)
    return ResponseFixtureConfig(
        deltas=deltas,
        coefficients=coefficients,
        baseline_positions_m=baseline,
        baseline_valid=np.ones((4, 2), dtype=bool),
        perturbed_valid=valid,
    )


def _plan(config: ResponseFixtureConfig) -> VariationPlan:
    specs = config.specs or (
        NoiseSpec(BALL_SPEED, scale=2.0, spec_id="input.ball-speed"),
        NoiseSpec(LAUNCH_ANGLE, scale=4.0, spec_id="input.launch-angle"),
    )
    return VariationPlan(
        mode="launch",
        base_variables={BALL_SPEED: 100.0, LAUNCH_ANGLE: 12.0},
        noise=specs,
        groups=config.groups,
        n_runs=len(TRIAL_IDS),
        seed=17,
    )


def _trace(
    plan: VariationPlan,
    inputs: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> EnsemblePositionTraces:
    dataset = VariationDataset(
        plan=plan,
        input_names=INPUT_KEYS,
        inputs=inputs,
        output_names=(),
        outputs=np.empty((len(TRIAL_IDS), 0)),
        success=np.ones(len(TRIAL_IDS), dtype=bool),
    )
    masked = np.where(valid[:, :, None, None], positions, np.nan)
    return EnsemblePositionTraces(
        variation=dataset,
        sample_times_s=np.array([0.0, 0.5]),
        coordinate_frame="swing.world",
        point_ids=POINT_IDS,
        positions_m=masked,
        sample_valid=valid,
        impact_sample_indices=np.full(len(TRIAL_IDS), -1),
    )


def build_response_inputs(
    config: ResponseFixtureConfig,
) -> tuple[ResponseFieldInput, ...]:
    """Build independently paired OAT inputs sharing one governed plan."""
    plan = _plan(config)
    metadata = make_execution_metadata(plan)
    bases = np.array([100.0, 12.0])
    baseline_inputs = np.broadcast_to(bases, (len(TRIAL_IDS), 2)).copy()
    baseline_trace = _trace(
        plan,
        baseline_inputs,
        config.baseline_positions_m,
        config.baseline_valid,
    )
    baseline = resample_position_traces(baseline_trace, baseline_trace.sample_times_s)
    result: list[ResponseFieldInput] = []
    for input_index, spec in enumerate(plan.noise):
        perturbed_inputs = baseline_inputs.copy()
        perturbed_inputs[:, input_index] += config.deltas[input_index]
        normalized = config.deltas[input_index] / spec.scale
        induced = np.einsum("t,spc->tspc", normalized, config.coefficients[input_index])
        positions = config.baseline_positions_m + induced
        perturbed_trace = _trace(
            plan,
            perturbed_inputs,
            positions,
            config.perturbed_valid[input_index],
        )
        perturbed = resample_position_traces(
            perturbed_trace, perturbed_trace.sample_times_s
        )
        result.append(
            ResponseFieldInput(
                spec_id=str(spec.spec_id),
                adapter_id="global_simulation_value/v1",
                source_layout_id="double_pendulum",
                trial_ids=TRIAL_IDS,
                baseline_trial_ids=TRIAL_IDS,
                perturbed_trial_ids=TRIAL_IDS,
                baseline=baseline,
                perturbed=perturbed,
                execution_metadata=metadata,
                source_sha256=SOURCE_SHA256,
                input_kind=config.input_kinds[input_index],
            )
        )
    return tuple(result)
