"""Construction of complete, trace-capable Rate simulation ensembles."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    BallSetup,
    BallSupportMode,
    SimulationConfig,
)
from rate_of_closure.variation.request_builder import (
    apply_global_simulation_values,
)
from rate_of_closure.variation.simulation_adapter import (
    build_simulation_ensemble_request,
)
from rate_of_closure.variation.simulation_types import SimulationEnsembleRequest
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.types import PendulumParameters
from shared.python.swing_sim.variation import (
    CATEGORY_BALL_SETUP,
    CATEGORY_CLUB,
    CATEGORY_DELIVERY,
    CATEGORY_SWING,
    NoiseSpec,
    VariationPlan,
)
from shared.python.swing_sim.variation.execution_metadata import (
    LEGACY_CURRENT_REGISTRY_WARNING,
    make_execution_metadata,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _key(category: str, name: str) -> str:
    return f"{category}.{name}"


_YAW = _key(CATEGORY_SWING, "yaw_deg")
_SIDE_TILT = _key(CATEGORY_SWING, "side_tilt_deg")
_FORWARD_TILT = _key(CATEGORY_SWING, "forward_tilt_deg")
_DAMPING_SHOULDER = _key(CATEGORY_SWING, "damping_shoulder")
_DAMPING_WRIST = _key(CATEGORY_SWING, "damping_wrist")
_SHOULDER_TORQUE_OFFSET = _key(CATEGORY_SWING, "shoulder_commanded_torque_offset_nm")
_WRIST_TORQUE_OFFSET = _key(CATEGORY_SWING, "wrist_commanded_torque_offset_nm")
_IMPACT_OFFSET = _key(CATEGORY_SWING, "impact_time_offset_s")
_TOE = _key(CATEGORY_DELIVERY, "impact_offset_toe_mm")
_HIGH = _key(CATEGORY_DELIVERY, "impact_offset_high_mm")
_HEAD_MASS = _key(CATEGORY_CLUB, "head_mass_kg")
_HEAD_MOI = _key(CATEGORY_CLUB, "head_moi_kg_m2")
_TEE = _key(CATEGORY_BALL_SETUP, "tee_height_m")
_DYNAMIC_LOFT = _key(CATEGORY_DELIVERY, "dynamic_loft_deg")


def _base_config() -> SimulationConfig:
    return SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=30.0),
        club=get_club("Driver 10.5°"),
        ball_setup=BallSetup(BallSupportMode.TEE, 0.0381),
        source_kind="double_pendulum",
        swing_duration_s=0.2,
    )


def _spec(variable_key: str, scale: float) -> NoiseSpec:
    return NoiseSpec(variable_key, distribution="uniform", scale=scale)


def _localized_spec(
    variable_key: str,
    point_id: str,
    *,
    window: tuple[float, float] = (0.02, 0.04),
    scale: float = 2.0,
) -> NoiseSpec:
    return NoiseSpec(
        variable_key,
        distribution="uniform",
        scale=scale,
        time_window_s=window,
        point_ids=(point_id,),
    )


def test_builder_maps_supported_samples_into_complete_configs() -> None:
    plan = VariationPlan(
        mode="swing",
        base_variables={
            _YAW: 2.0,
            _DAMPING_SHOULDER: 0.4,
            _DAMPING_WRIST: 0.25,
            _IMPACT_OFFSET: 0.0,
            _TOE: 1.0,
            _HIGH: -2.0,
            _HEAD_MASS: 0.2,
            _HEAD_MOI: 4.5e-4,
            _TEE: 0.0381,
        },
        noise=(
            _spec(_YAW, 1.0),
            _spec(_DAMPING_SHOULDER, 0.02),
            _spec(_DAMPING_WRIST, 0.02),
            _spec(_IMPACT_OFFSET, 0.004),
            _spec(_TOE, 2.0),
            _spec(_HIGH, 2.0),
            _spec(_HEAD_MASS, 0.002),
            _spec(_HEAD_MOI, 1e-5),
            _spec(_TEE, 0.002),
        ),
        n_runs=3,
        seed=17,
    )

    request = build_simulation_ensemble_request(plan, _base_config())

    assert request.sampled_inputs.shape == (3, len(plan.noise))
    for row, config in zip(request.sampled_inputs, request.configs, strict=True):
        values = dict(zip((spec.variable_key for spec in plan.noise), row, strict=True))
        assert config.plane.yaw_deg == pytest.approx(values[_YAW])
        assert config.pendulum_parameters is not None
        assert config.pendulum_parameters.d1 == pytest.approx(values[_DAMPING_SHOULDER])
        assert config.pendulum_parameters.d2 == pytest.approx(values[_DAMPING_WRIST])
        assert config.impact_time_offset_s == pytest.approx(values[_IMPACT_OFFSET])
        assert config.scenario.impact_offset_toe_mm == pytest.approx(values[_TOE])
        assert config.scenario.impact_offset_high_mm == pytest.approx(values[_HIGH])
        assert config.club.head_mass_kg == pytest.approx(values[_HEAD_MASS])
        assert config.club.moi_about_shaft_kg_m2 == pytest.approx(values[_HEAD_MOI])
        assert config.ball_setup.tee_height_m == pytest.approx(values[_TEE])


def test_builder_uses_default_pendulum_parameters_when_not_explicit() -> None:
    plan = VariationPlan(
        mode="swing",
        noise=(_spec(_YAW, 0.1),),
        n_runs=2,
        seed=3,
    )

    request = build_simulation_ensemble_request(plan, _base_config())

    expected = PendulumParameters.golf_default()
    assert all(config.pendulum_parameters == expected for config in request.configs)


def test_builder_binds_fresh_execution_metadata_without_legacy_warning() -> None:
    plan = VariationPlan(mode="swing", noise=(_spec(_YAW, 0.1),), n_runs=2)

    request = build_simulation_ensemble_request(plan, _base_config())

    assert request.execution_metadata == make_execution_metadata(plan)
    assert request.metadata_warning is None


def test_direct_legacy_request_resolves_current_registry_with_warning() -> None:
    plan = VariationPlan(mode="swing", noise=(_spec(_YAW, 0.1),), n_runs=2)
    built = build_simulation_ensemble_request(plan, _base_config())

    legacy = SimulationEnsembleRequest(plan, built.sampled_inputs, built.configs)

    assert legacy.execution_metadata == make_execution_metadata(plan)
    assert legacy.metadata_warning == LEGACY_CURRENT_REGISTRY_WARNING


def test_request_rejects_cross_plan_execution_metadata() -> None:
    first = VariationPlan(mode="swing", noise=(_spec(_YAW, 0.1),), n_runs=2)
    second = dataclasses.replace(first, seed=first.seed + 1)
    built = build_simulation_ensemble_request(second, _base_config())

    with pytest.raises(ContractViolationError, match="plan digest"):
        SimulationEnsembleRequest(
            second,
            built.sampled_inputs,
            built.configs,
            execution_metadata=make_execution_metadata(first),
        )


@pytest.mark.parametrize("mutation", ["value", "permutation", "subset", "config_order"])
def test_identity_request_rejects_sample_or_config_order_drift(mutation: str) -> None:
    plan = VariationPlan(mode="swing", noise=(_spec(_YAW, 0.1),), n_runs=3, seed=8)
    built = build_simulation_ensemble_request(plan, _base_config())
    samples = np.array(built.sampled_inputs, copy=True)
    configs = built.configs
    if mutation == "value":
        samples[0, 0] += 0.01
    elif mutation == "permutation":
        samples = samples[::-1]
    elif mutation == "subset":
        samples = samples[:-1]
        configs = configs[:-1]
    else:
        configs = configs[::-1]

    with pytest.raises(
        ContractViolationError, match="sampled_inputs|config order|configs must contain"
    ):
        SimulationEnsembleRequest(plan, samples, configs, built.execution_metadata)


def test_global_value_seam_applies_every_fixed_contact_morris_variable() -> None:
    values = {
        _YAW: 1.0,
        _SIDE_TILT: -42.0,
        _FORWARD_TILT: 3.0,
        _DAMPING_SHOULDER: 0.41,
        _DAMPING_WRIST: 0.24,
        _TOE: 2.0,
        _HIGH: -1.0,
        _HEAD_MASS: 0.201,
        _HEAD_MOI: 4.6e-4,
        _TEE: 0.04,
    }

    config = apply_global_simulation_values(_base_config(), values)

    assert dataclasses.asdict(config.plane) == {
        "yaw_deg": 1.0,
        "side_tilt_deg": -42.0,
        "forward_tilt_deg": 3.0,
    }
    assert config.pendulum_parameters.d1 == pytest.approx(0.41)
    assert config.pendulum_parameters.d2 == pytest.approx(0.24)
    assert config.scenario.impact_offset_toe_mm == pytest.approx(2.0)
    assert config.scenario.impact_offset_high_mm == pytest.approx(-1.0)
    assert config.club.head_mass_kg == pytest.approx(0.201)
    assert config.club.moi_about_shaft_kg_m2 == pytest.approx(4.6e-4)
    assert config.ball_setup.tee_height_m == pytest.approx(0.04)


@pytest.mark.parametrize("value", [True, "1.0", float("inf")])
def test_global_value_seam_rejects_coercive_or_nonfinite_values(value: object) -> None:
    with pytest.raises(ContractViolationError, match="real scalars|finite"):
        apply_global_simulation_values(_base_config(), {_YAW: value})  # type: ignore[dict-item]


def test_builder_rejects_unsupported_localized_or_unmapped_variables() -> None:
    localized = VariationPlan(
        mode="swing",
        noise=(
            dataclasses.replace(
                _spec(_YAW, 0.1),
                time_window_s=(0.1, 0.2),
            ),
        ),
        n_runs=2,
    )
    unsupported = VariationPlan(
        mode="swing",
        noise=(_spec(_DYNAMIC_LOFT, 0.5),),
        n_runs=2,
    )

    with pytest.raises(ContractViolationError, match="localized"):
        build_simulation_ensemble_request(localized, _base_config())
    with pytest.raises(ContractViolationError, match="not trace-capable"):
        build_simulation_ensemble_request(unsupported, _base_config())


def test_builder_requires_swing_mode_and_double_pendulum_source() -> None:
    delivery = VariationPlan(
        mode="delivery",
        noise=(_spec(_TOE, 1.0),),
        n_runs=2,
    )
    manual = dataclasses.replace(_base_config(), source_kind="manual")

    with pytest.raises(ContractViolationError, match="swing mode"):
        build_simulation_ensemble_request(delivery, _base_config())
    with pytest.raises(ContractViolationError, match="double_pendulum"):
        build_simulation_ensemble_request(
            VariationPlan(mode="swing", noise=(_spec(_YAW, 0.1),), n_runs=2),
            manual,
        )
