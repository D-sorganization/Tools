"""Rate-specific execution adapter for shared Morris designs (#4142 R13.3)."""

from __future__ import annotations

import dataclasses
import threading

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    BallSetup,
    BallSupportMode,
    ContactMode,
    SimulationConfig,
    SimulationRun,
    run_simulation,
)
from rate_of_closure.variation.morris_rate_adapter import (
    RATE_MORRIS_OUTPUTS,
    RATE_MORRIS_VARIABLE_KEYS,
    RateMorrisEvaluator,
    evaluate_rate_morris_design,
)
from rate_of_closure.variation.simulation_types import ALL_OUTPUT_NAMES, APP_FRAME_ID
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_BALL_SETUP,
    CATEGORY_CLUB,
    CATEGORY_DELIVERY,
    CATEGORY_SWING,
    CancelledError,
    MorrisDesign,
    MorrisExecutionOptions,
    MorrisFactor,
    MorrisSample,
    evaluate_morris_design,
    generate_morris_design,
    variable_registry,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _key(category: str, name: str) -> str:
    return f"{category}.{name}"


_YAW = _key(CATEGORY_SWING, "yaw_deg")
_SIDE = _key(CATEGORY_SWING, "side_tilt_deg")
_FORWARD = _key(CATEGORY_SWING, "forward_tilt_deg")
_DAMPING_SHOULDER = _key(CATEGORY_SWING, "damping_shoulder")
_DAMPING_WRIST = _key(CATEGORY_SWING, "damping_wrist")
_IMPACT_TIME_OFFSET = _key(CATEGORY_SWING, "impact_time_offset_s")
_TOE = _key(CATEGORY_DELIVERY, "impact_offset_toe_mm")
_HIGH = _key(CATEGORY_DELIVERY, "impact_offset_high_mm")
_HEAD_MASS = _key(CATEGORY_CLUB, "head_mass_kg")
_HEAD_MOI = _key(CATEGORY_CLUB, "head_moi_kg_m2")
_TEE = _key(CATEGORY_BALL_SETUP, "tee_height_m")
_DRIVER = get_club("Driver 10.5°")


def _base_config() -> SimulationConfig:
    return SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=113.0),
        club=_DRIVER,
        ball_setup=BallSetup(BallSupportMode.TEE, 0.0381),
        source_kind="double_pendulum",
        swing_duration_s=0.05,
        contact_mode=ContactMode.FIXED_BALL_CONTACT,
    )


def _factor(
    variable_key: str,
    *,
    spec_id: str | None = None,
    unit: str | None = None,
) -> MorrisFactor:
    definition = variable_registry()[variable_key]
    return MorrisFactor(
        spec_id=spec_id or variable_key.rsplit(".", 1)[-1],
        variable_key=variable_key,
        lower=definition.default - definition.typical_scale,
        upper=definition.default + definition.typical_scale,
        unit=definition.unit if unit is None else unit,
    )


def _design(*factors: MorrisFactor, trajectories: int = 2) -> MorrisDesign:
    selected = factors or (_factor(_YAW), _factor(_TEE))
    return generate_morris_design(tuple(selected), trajectories=trajectories, seed=17)


def test_rate_output_metadata_matches_exact_current_scalar_contract() -> None:
    app_frame_names = {
        "spin_axis_tilt_deg",
        "launch_angle_deg",
        "launch_azimuth_deg",
        "carry_m",
        "lateral_m",
        "max_height_m",
        "landing_angle_deg",
    }
    units = (
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

    assert tuple(output.name for output in RATE_MORRIS_OUTPUTS) == ALL_OUTPUT_NAMES
    assert tuple(output.unit for output in RATE_MORRIS_OUTPUTS) == units
    assert tuple(output.target_kind for output in RATE_MORRIS_OUTPUTS) == (
        *("scalar",) * 3,
        *("impact",) * 5,
        *("shot-outcome",) * 9,
    )
    assert {
        output.name for output in RATE_MORRIS_OUTPUTS if output.coordinate_frame
    } == app_frame_names
    assert all(
        output.coordinate_frame == APP_FRAME_ID
        for output in RATE_MORRIS_OUTPUTS
        if output.name in app_frame_names
    )


def test_rate_variable_contract_is_the_exact_supported_global_set() -> None:
    assert RATE_MORRIS_VARIABLE_KEYS == {
        _YAW,
        _SIDE,
        _FORWARD,
        _DAMPING_SHOULDER,
        _DAMPING_WRIST,
        _TOE,
        _HIGH,
        _HEAD_MASS,
        _HEAD_MOI,
        _TEE,
    }


def test_sample_maps_spec_ids_to_supported_simulation_variable_keys() -> None:
    design = _design(
        _factor(_YAW, spec_id="plane-yaw"),
        _factor(_DAMPING_SHOULDER, spec_id="shoulder-loss"),
        _factor(_TOE, spec_id="toe-strike"),
        _factor(_HEAD_MASS, spec_id="head-mass"),
        _factor(_TEE, spec_id="tee-height"),
        trajectories=1,
    )
    seen: list[SimulationConfig] = []

    def executor(config: SimulationConfig) -> SimulationRun:
        seen.append(config)
        return run_simulation(config)

    evaluator = RateMorrisEvaluator(design, _base_config(), executor)
    point = design.physical_points[0, 0]
    sample = MorrisSample(
        ordinal=0,
        trajectory_index=0,
        point_index=0,
        factors=design.factors,
        physical_values=dict(
            zip((factor.spec_id for factor in design.factors), point, strict=True)
        ),
    )

    evaluation = evaluator(sample)
    config = seen[0]

    assert evaluation.status == "evaluated_no_impact"
    assert config.plane.yaw_deg == pytest.approx(point[0])
    assert config.pendulum_parameters.d1 == pytest.approx(point[1])
    assert config.scenario.impact_offset_toe_mm == pytest.approx(point[2])
    assert config.club.head_mass_kg == pytest.approx(point[3])
    assert config.ball_setup.tee_height_m == pytest.approx(point[4])


def test_injected_hit_miss_and_failure_preserve_exact_availability() -> None:
    design = _design(_factor(_YAW), trajectories=1)
    manual_hit = run_simulation(
        dataclasses.replace(
            _base_config(),
            source_kind="manual",
            ball_setup=BallSetup(BallSupportMode.GROUND),
        )
    )
    fixed_miss = run_simulation(_base_config())
    runs = iter((manual_hit, fixed_miss))

    hit_or_miss = RateMorrisEvaluator(
        design, _base_config(), lambda _config: next(runs)
    )
    first = hit_or_miss(_sample(design, 0))
    second = hit_or_miss(_sample(design, 1))

    assert first.status == "evaluated_hit"
    assert all(first.values[name] is not None for name in ALL_OUTPUT_NAMES)
    assert second.status == "evaluated_no_impact"
    assert all(second.values[name] is not None for name in ALL_OUTPUT_NAMES[:3])
    assert all(second.values[name] is None for name in ALL_OUTPUT_NAMES[3:])

    def failure(_config: SimulationConfig) -> SimulationRun:
        raise FloatingPointError("planted numerical failure")

    failed = RateMorrisEvaluator(design, _base_config(), failure)(_sample(design, 0))
    assert failed.status == "numerical_failure"
    assert all(value is None for value in failed.values.values())


def _sample(design: MorrisDesign, ordinal: int) -> MorrisSample:
    points_per_trajectory = len(design.factors) + 1
    trajectory, point_index = divmod(ordinal, points_per_trajectory)
    point = design.physical_points[trajectory, point_index]
    return MorrisSample(
        ordinal,
        trajectory,
        point_index,
        design.factors,
        dict(zip((factor.spec_id for factor in design.factors), point, strict=True)),
    )


def test_full_rate_execution_records_genuine_double_pendulum_fixed_miss() -> None:
    observations = evaluate_rate_morris_design(_design(_factor(_YAW)), _base_config())

    assert np.all(observations.outcomes == "evaluated_no_impact")
    assert np.all(np.isfinite(observations.values[:, :, :3]))
    assert np.all(np.isnan(observations.values[:, :, 3:]))


def test_rate_execution_honors_shared_cooperative_cancellation() -> None:
    cancellation = threading.Event()
    calls = 0

    def executor(config: SimulationConfig) -> SimulationRun:
        nonlocal calls
        calls += 1
        return run_simulation(config)

    cancellation.set()
    with pytest.raises(CancelledError, match="before start"):
        evaluate_rate_morris_design(
            _design(_factor(_YAW)),
            _base_config(),
            MorrisExecutionOptions(cancel_event=cancellation),
            executor,
        )
    assert calls == 0


@pytest.mark.parametrize("source_kind", ["manual", "triple_pendulum"])
def test_evaluator_requires_double_pendulum_source(source_kind: str) -> None:
    with pytest.raises(ContractViolationError, match="double_pendulum"):
        RateMorrisEvaluator(
            _design(_factor(_YAW)),
            dataclasses.replace(_base_config(), source_kind=source_kind),
        )


def test_evaluator_requires_fixed_ball_contact() -> None:
    with pytest.raises(ContractViolationError, match="FIXED_BALL_CONTACT"):
        RateMorrisEvaluator(
            _design(_factor(_YAW)),
            dataclasses.replace(
                _base_config(), contact_mode=ContactMode.DELIVERY_INSPECTION
            ),
        )


def test_evaluator_rejects_localized_duplicate_and_wrong_unit_factors() -> None:
    localized = dataclasses.replace(_factor(_YAW), source_time_window_s=(0.0, 0.01))
    localized_point = dataclasses.replace(
        _factor(_SIDE), source_point_ids=("swing.clubhead.reference",)
    )
    duplicate = _factor(_YAW, spec_id="another-yaw")
    wrong_unit = _factor(_YAW)
    object.__setattr__(wrong_unit, "unit", "rad")

    with pytest.raises(ContractViolationError, match="global"):
        RateMorrisEvaluator(_design(localized), _base_config())
    with pytest.raises(ContractViolationError, match="global"):
        RateMorrisEvaluator(_design(localized_point), _base_config())
    with pytest.raises(ContractViolationError, match="unique variable_key"):
        RateMorrisEvaluator(_design(_factor(_YAW), duplicate), _base_config())
    with pytest.raises(ContractViolationError, match="registered unit"):
        RateMorrisEvaluator(_design(wrong_unit), _base_config())


@pytest.mark.parametrize(
    "variable_key",
    [
        _IMPACT_TIME_OFFSET,
        _key(CATEGORY_DELIVERY, "dynamic_loft_deg"),
    ],
)
def test_evaluator_rejects_noop_or_unsupported_variables(variable_key: str) -> None:
    message = (
        "fixed-ball contact ignores"
        if variable_key == _IMPACT_TIME_OFFSET
        else "supported"
    )
    with pytest.raises(ContractViolationError, match=message):
        RateMorrisEvaluator(_design(_factor(variable_key)), _base_config())


def test_tee_height_requires_tee_support() -> None:
    ground = dataclasses.replace(
        _base_config(), ball_setup=BallSetup(BallSupportMode.GROUND)
    )
    with pytest.raises(ContractViolationError, match="Tee support"):
        RateMorrisEvaluator(_design(_factor(_TEE)), ground)


def test_evaluator_rejects_sample_or_output_contract_drift() -> None:
    design = _design(_factor(_YAW), _factor(_SIDE), trajectories=1)
    evaluator = RateMorrisEvaluator(design, _base_config())
    wrong_design = _design(_factor(_FORWARD), trajectories=1)

    with pytest.raises(ContractViolationError, match="design factors"):
        evaluator(_sample(wrong_design, 0))
    with pytest.raises(ContractViolationError, match="exact output-name set"):
        evaluate_morris_design(design, RATE_MORRIS_OUTPUTS[:-1], evaluator)

    malformed = _sample(design, 0)
    object.__setattr__(malformed, "ordinal", 1)
    with pytest.raises(ContractViolationError, match="identity must agree"):
        evaluator(malformed)


def test_programming_defect_propagates() -> None:
    def broken(_config: SimulationConfig) -> SimulationRun:
        raise TypeError("programming defect")

    evaluator = RateMorrisEvaluator(_design(_factor(_YAW)), _base_config(), broken)
    with pytest.raises(TypeError, match="programming defect"):
        evaluator(_sample(evaluator.design, 0))
