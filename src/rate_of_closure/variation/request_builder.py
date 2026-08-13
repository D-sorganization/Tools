"""Build complete Rate simulation requests from shared variation plans."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from numbers import Real
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from rate_of_closure.simulation import BallSetup, BallSupportMode, SimulationConfig
from rate_of_closure.variation.simulation_types import SimulationEnsembleRequest
from shared.python.contracts import require
from shared.python.swing_sim.integration_grid import effective_rk4_duration
from shared.python.swing_sim.run_config import (
    LocalizedTorqueOffset,
)
from shared.python.swing_sim.types import PlaneOrientation
from shared.python.swing_sim.variation import (
    CATEGORY_BALL_SETUP,
    CATEGORY_CLUB,
    CATEGORY_DELIVERY,
    CATEGORY_SWING,
    LOCALIZED_TORQUE_VARIABLE_JOINTS,
    VariationPlan,
    sample_inputs,
)


def _key(category: str, name: str) -> str:
    return f"{category}.{name}"


_YAW = _key(CATEGORY_SWING, "yaw_deg")
_SIDE_TILT = _key(CATEGORY_SWING, "side_tilt_deg")
_FORWARD_TILT = _key(CATEGORY_SWING, "forward_tilt_deg")
_IMPACT_TIME_OFFSET = _key(CATEGORY_SWING, "impact_time_offset_s")
_DAMPING_SHOULDER = _key(CATEGORY_SWING, "damping_shoulder")
_DAMPING_WRIST = _key(CATEGORY_SWING, "damping_wrist")
_TOE_OFFSET = _key(CATEGORY_DELIVERY, "impact_offset_toe_mm")
_HIGH_OFFSET = _key(CATEGORY_DELIVERY, "impact_offset_high_mm")
_HEAD_MASS = _key(CATEGORY_CLUB, "head_mass_kg")
_HEAD_MOI = _key(CATEGORY_CLUB, "head_moi_kg_m2")
_TEE_HEIGHT = _key(CATEGORY_BALL_SETUP, "tee_height_m")

GLOBAL_TRACE_VARIABLE_KEYS = frozenset(
    {
        _YAW,
        _SIDE_TILT,
        _FORWARD_TILT,
        _IMPACT_TIME_OFFSET,
        _DAMPING_SHOULDER,
        _DAMPING_WRIST,
        _TOE_OFFSET,
        _HIGH_OFFSET,
        _HEAD_MASS,
        _HEAD_MOI,
        _TEE_HEIGHT,
    }
)

TRACE_CAPABLE_VARIABLE_KEYS = frozenset(
    GLOBAL_TRACE_VARIABLE_KEYS | set(LOCALIZED_TORQUE_VARIABLE_JOINTS)
)


def _is_real_scalar(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_))


def build_simulation_ensemble_request(
    plan: VariationPlan,
    base_config: SimulationConfig,
) -> SimulationEnsembleRequest:
    """Sample ``plan`` and map every row to a complete simulation config.

    The adapter deliberately rejects variables whose full simulation effect
    is not modeled. This prevents an arc plot from implying that a scalar-only
    delivery perturbation changed the swing geometry.
    """
    _validate_request_context(plan, base_config)
    samples = sample_inputs(plan)
    return build_simulation_ensemble_request_from_samples(plan, base_config, samples)


def _validate_request_context(
    plan: VariationPlan, base_config: SimulationConfig
) -> None:
    """Validate the common plan/config capability boundary once."""
    require(isinstance(plan, VariationPlan), "plan must be a VariationPlan")
    require(
        isinstance(base_config, SimulationConfig),
        "base_config must be a SimulationConfig",
    )
    require(plan.mode == "swing", "trace ensembles require swing mode", plan.mode)
    require(
        base_config.source_kind == "double_pendulum",
        "trace ensembles currently require the double_pendulum source",
        base_config.source_kind,
    )
    requested = {spec.variable_key for spec in plan.noise} | set(plan.base_variables)
    unsupported = sorted(requested - TRACE_CAPABLE_VARIABLE_KEYS)
    require(not unsupported, "variables are not trace-capable", unsupported)


def _normalize_explicit_samples(
    sampled_inputs: np.ndarray, plan: VariationPlan
) -> npt.NDArray[np.float64]:
    """Own and validate an explicit design matrix before config allocation."""
    raw_samples = np.asarray(sampled_inputs)
    require(
        np.issubdtype(raw_samples.dtype, np.number)
        and not np.issubdtype(raw_samples.dtype, np.bool_)
        and not np.issubdtype(raw_samples.dtype, np.complexfloating),
        "sampled_inputs must contain real non-boolean numbers",
    )
    samples: npt.NDArray[np.float64] = np.array(
        raw_samples, dtype=np.float64, copy=True
    )
    require(
        samples.shape == (plan.n_runs, len(plan.noise)),
        "sampled_inputs has invalid shape",
        samples.shape,
    )
    require(bool(np.all(np.isfinite(samples))), "sampled_inputs must be finite")
    return samples


def build_simulation_ensemble_request_from_samples(
    plan: VariationPlan,
    base_config: SimulationConfig,
    sampled_inputs: np.ndarray,
) -> SimulationEnsembleRequest:
    """Build a request from an explicit finite sample matrix.

    This seam is for deterministic experimental designs whose rows are the
    scientific authority (for example, planted baseline/perturbation pairs),
    rather than pseudorandom Monte Carlo draws.
    """
    _validate_request_context(plan, base_config)
    samples = _normalize_explicit_samples(sampled_inputs, plan)
    _validate_noise_loci(plan, base_config)
    configs = tuple(
        _apply_row(base_config, plan, row) for row in np.asarray(samples, dtype=float)
    )
    return SimulationEnsembleRequest(plan, samples, configs)


def _apply_row(
    base: SimulationConfig,
    plan: VariationPlan,
    row: np.ndarray,
) -> SimulationConfig:
    """Apply one sampled row plus explicit plan bases to ``base``."""
    localized_keys = set(LOCALIZED_TORQUE_VARIABLE_JOINTS)
    values = {
        key: value
        for key, value in plan.base_variables.items()
        if key not in localized_keys
    }
    offsets: list[LocalizedTorqueOffset] = []
    for spec, sampled_value in zip(plan.noise, row, strict=True):
        if spec.variable_key not in localized_keys:
            values[spec.variable_key] = float(sampled_value)
            continue
        offsets.append(
            LocalizedTorqueOffset(
                joint_id=LOCALIZED_TORQUE_VARIABLE_JOINTS[spec.variable_key],
                time_window_s=spec.time_window_s or (0.0, 0.0),
                torque_nm=float(sampled_value),
            )
        )
    updated = apply_global_simulation_values(base, values)
    run_config = updated.swing_run_config
    return replace(
        updated,
        swing_run_config=replace(
            run_config,
            commanded_torque_offsets=run_config.commanded_torque_offsets
            + tuple(offsets),
        ),
    )


def _validate_noise_loci(plan: VariationPlan, base_config: SimulationConfig) -> None:
    """Validate global variables and exact localized torque loci."""
    localized_specs = {
        spec.variable_key: spec
        for spec in plan.noise
        if spec.variable_key in LOCALIZED_TORQUE_VARIABLE_JOINTS
    }
    effective_duration_s = (
        effective_rk4_duration(base_config.swing_duration_s)
        if localized_specs
        else base_config.swing_duration_s
    )
    base_only = (
        set(plan.base_variables) & set(LOCALIZED_TORQUE_VARIABLE_JOINTS)
    ) - set(localized_specs)
    require(
        not base_only,
        "localized torque base variables require a matching noise specification",
        sorted(base_only),
    )
    for spec in plan.noise:
        expected_joint = LOCALIZED_TORQUE_VARIABLE_JOINTS.get(spec.variable_key)
        if expected_joint is None:
            require(
                spec.is_global,
                "localized perturbation is unsupported for this variable",
                spec.spec_id,
            )
            continue
        window = spec.time_window_s
        require(
            window is not None,
            "localized torque perturbation requires time_window_s",
            spec.spec_id,
        )
        require(
            spec.point_ids == (expected_joint,),
            "localized torque perturbation requires its exact topological joint point",
            (spec.point_ids, expected_joint),
        )
        assert window is not None
        start_s, end_s = window
        require(
            0.0 <= start_s < end_s <= effective_duration_s,
            "localized torque time window must lie within the effective RK4 duration",
            (window, effective_duration_s),
        )


def apply_global_simulation_values(
    config: SimulationConfig, values: Mapping[str, float]
) -> SimulationConfig:
    """Apply exact supported global variation values to one Rate config.

    Preconditions: keys are trace-capable and values are finite. Contextual tee
    support is validated before returning a new immutable configuration.
    """
    require(isinstance(config, SimulationConfig), "config must be SimulationConfig")
    require(isinstance(values, Mapping), "values must be a mapping")
    require(
        all(isinstance(key, str) for key in values),
        "global simulation value keys must be strings",
    )
    unsupported = sorted(set(values) - GLOBAL_TRACE_VARIABLE_KEYS)
    require(not unsupported, "variables are not trace-capable", unsupported)
    require(
        all(_is_real_scalar(value) for value in values.values()),
        "global simulation values must be real scalars",
    )
    normalized = {key: float(value) for key, value in values.items()}
    require(
        all(math.isfinite(value) for value in normalized.values()),
        "global simulation values must be finite",
    )
    updated = _apply_plane_and_dynamics(config, normalized)
    updated = _apply_scenario(updated, normalized)
    updated = _apply_club(updated, normalized)
    return _apply_tee(updated, normalized)


def _apply_plane_and_dynamics(
    config: SimulationConfig,
    values: dict[str, float],
) -> SimulationConfig:
    """Apply plane, timing, and passive double-pendulum damping values."""
    plane = PlaneOrientation(
        yaw_deg=values.get(_YAW, config.plane.yaw_deg),
        side_tilt_deg=values.get(_SIDE_TILT, config.plane.side_tilt_deg),
        forward_tilt_deg=values.get(_FORWARD_TILT, config.plane.forward_tilt_deg),
    )
    parameters = replace(
        config.pendulum_parameters,
        d1=values.get(_DAMPING_SHOULDER, config.pendulum_parameters.d1),
        d2=values.get(_DAMPING_WRIST, config.pendulum_parameters.d2),
    )
    return replace(
        config,
        plane=plane,
        impact_time_offset_s=values.get(
            _IMPACT_TIME_OFFSET, config.impact_time_offset_s
        ),
        pendulum_parameters=parameters,
    )


def _apply_scenario(
    config: SimulationConfig,
    values: dict[str, float],
) -> SimulationConfig:
    """Apply delivery values owned by the manual/impact scenario record."""
    scenario = replace(
        config.scenario,
        impact_offset_toe_mm=values.get(
            _TOE_OFFSET, config.scenario.impact_offset_toe_mm
        ),
        impact_offset_high_mm=values.get(
            _HIGH_OFFSET, config.scenario.impact_offset_high_mm
        ),
    )
    return replace(config, scenario=scenario)


def _apply_club(
    config: SimulationConfig,
    values: dict[str, float],
) -> SimulationConfig:
    """Apply inertial variables represented by the canonical ClubSpec."""
    club = replace(
        config.club,
        head_mass_kg=values.get(_HEAD_MASS, config.club.head_mass_kg),
        moi_about_shaft_kg_m2=values.get(_HEAD_MOI, config.club.moi_about_shaft_kg_m2),
    )
    return replace(config, club=club)


def _apply_tee(
    config: SimulationConfig,
    values: dict[str, float],
) -> SimulationConfig:
    """Apply a tee-height value only when the declared support is Tee."""
    if _TEE_HEIGHT not in values:
        return config
    require(
        config.ball_setup.support_mode is BallSupportMode.TEE,
        "tee_height_m variation requires Tee support",
        config.ball_setup.support_mode,
    )
    return replace(
        config,
        ball_setup=BallSetup(BallSupportMode.TEE, values[_TEE_HEIGHT]),
    )


TRACE_CAPABILITIES = MappingProxyType(
    {
        "mode": "swing",
        "source_kind": "double_pendulum",
        "localized_torque_offsets": tuple(LOCALIZED_TORQUE_VARIABLE_JOINTS.items()),
    }
)

__all__ = [
    "TRACE_CAPABILITIES",
    "TRACE_CAPABLE_VARIABLE_KEYS",
    "LOCALIZED_TORQUE_VARIABLE_JOINTS",
    "apply_global_simulation_values",
    "build_simulation_ensemble_request",
    "build_simulation_ensemble_request_from_samples",
]
