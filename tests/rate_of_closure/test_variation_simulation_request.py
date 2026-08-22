"""Construction of complete, trace-capable Rate simulation ensembles."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    BallSetup,
    BallSupportMode,
    SimulationConfig,
)
from rate_of_closure.variation._simulation_config_identity import (
    simulation_configurations_sha256,
)
from rate_of_closure.variation.ensemble_source import SimulationEnsembleSource
from rate_of_closure.variation.request_builder import (
    apply_global_simulation_values,
)
from rate_of_closure.variation.simulation_adapter import (
    build_ensemble_stream_header,
    build_simulation_ensemble_request,
)
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

    assert isinstance(request, SimulationEnsembleSource)
    assert not hasattr(request, "sampled_inputs")
    assert not hasattr(request, "configs")
    header = build_ensemble_stream_header(request)
    assert header.sampled_inputs is None
    assert len(header.sampled_input_sha256) == 64
    chunks = tuple(request.work_chunks(chunk_size=2, start_index=0))
    assert tuple(chunk.start_index for chunk in chunks) == (0, 2)
    sampled_inputs = np.vstack([chunk.sampled_inputs for chunk in chunks])
    configs = tuple(config for chunk in chunks for config in chunk.configs)
    assert (
        header.sampled_input_sha256
        == hashlib.sha256(sampled_inputs.tobytes(order="C")).hexdigest()
    )
    assert header.configuration_sha256 == simulation_configurations_sha256(configs)
    for row, config in zip(sampled_inputs, configs, strict=True):
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
    configs = tuple(
        config
        for chunk in request.work_chunks(chunk_size=1, start_index=0)
        for config in chunk.configs
    )
    assert all(config.pendulum_parameters == expected for config in configs)


def test_lazy_request_resume_generates_only_the_requested_suffix() -> None:
    plan = VariationPlan(
        mode="swing",
        noise=(_spec(_YAW, 0.1),),
        n_runs=7,
        seed=3,
    )
    request = build_simulation_ensemble_request(plan, _base_config())

    chunks = tuple(request.work_chunks(chunk_size=2, start_index=5))

    assert tuple(chunk.start_index for chunk in chunks) == (5, 6)
    assert sum(len(chunk.configs) for chunk in chunks) == 2


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
