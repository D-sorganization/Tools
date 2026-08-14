"""Rate run-history contracts for applied joint torques and profile export."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    ContactMode,
    ImpactStatus,
    SimulationConfig,
    compute_kinetics,
    fit_run_torque_profile,
    run_simulation,
    run_to_json_dict,
    write_torque_csv,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.run_config import (
    DOUBLE_PENDULUM_MODEL_ID,
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
    DoublePendulumRunConfig,
    JointLockConfig,
    LocalizedTorqueOffset,
)
from shared.python.swing_sim.torque_library import TorqueProfileLibrary
from shared.python.swing_sim.torque_profiles import (
    JointTorqueAssignment,
    PrescribedTorqueProfile,
    TorquePolynomial,
    TorqueProfileSource,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SCENARIO = ImpactScenario(clubhead_speed_mph=113.0)
_DRIVER = get_club("Driver 10.5°")
_DURATION_S = 0.05


def _prescribed_profile(
    profile_id: str = "profile.rate.constant.v1",
) -> PrescribedTorqueProfile:
    return PrescribedTorqueProfile(
        profile_id=profile_id,
        model_id=DOUBLE_PENDULUM_MODEL_ID,
        name="Rate Constant Torque",
        description="Constant torques used to verify retained run histories.",
        source=TorqueProfileSource.DIRECT,
        source_metadata={"author": "rate-test-suite"},
        created_at_utc="2026-08-05T12:00:00Z",
        modified_at_utc="2026-08-05T12:00:00Z",
        time_domain_s=(0.0, _DURATION_S),
        assignments=(
            JointTorqueAssignment(SHOULDER_JOINT_ID, TorquePolynomial((12.0,))),
            JointTorqueAssignment(WRIST_JOINT_ID, TorquePolynomial((-3.0,))),
        ),
    )


def _prescribed_miss(profile: PrescribedTorqueProfile | None = None):  # type: ignore[no-untyped-def]
    selected = profile or _prescribed_profile()
    return run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=_DRIVER,
            source_kind="double_pendulum",
            swing_duration_s=_DURATION_S,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
            swing_run_config=DoublePendulumRunConfig.prescribed(selected.profile_id),
            torque_library=TorqueProfileLibrary((selected,)),
        )
    )


def test_passive_double_pendulum_retains_stable_zero_torque_history() -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=_DRIVER,
            source_kind="double_pendulum",
            swing_duration_s=_DURATION_S,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
        )
    )
    assert run.swing_joint_ids == (SHOULDER_JOINT_ID, WRIST_JOINT_ID)
    assert run.swing_applied_torques_nm.shape == (len(run.swing_times), 2)
    np.testing.assert_array_equal(run.swing_applied_torques_nm, 0.0)


def test_unsupported_manual_source_retains_explicit_empty_history() -> None:
    run = run_simulation(SimulationConfig(scenario=_SCENARIO, club=_DRIVER))
    assert run.swing_joint_ids == ()
    assert run.swing_applied_torques_nm.shape == (len(run.swing_times), 0)
    with pytest.raises(ContractViolationError, match="no applied joint torque"):
        fit_run_torque_profile(
            run,
            profile_id="profile.invalid.manual",
            name="Invalid Manual Export",
            description="Manual sources do not expose joint torques.",
            degree=1,
            source_metadata={"run_id": "run.manual"},
            created_at_utc="2026-08-05T12:00:00Z",
            modified_at_utc="2026-08-05T12:00:00Z",
        )


def test_unsupported_source_torque_csv_is_a_header_only_export(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    run = run_simulation(SimulationConfig(scenario=_SCENARIO, club=_DRIVER))
    path = tmp_path / "manual-torques.csv"
    write_torque_csv(run, path)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "t_s,joint_id,applied_torque_nm"
    ]


def test_prescribed_miss_retains_history_and_exports_replayable_profile() -> None:
    run = _prescribed_miss()
    assert run.impact_outcome.status is ImpactStatus.MISS
    assert run.swing_joint_ids == (SHOULDER_JOINT_ID, WRIST_JOINT_ID)
    np.testing.assert_allclose(
        run.swing_applied_torques_nm,
        np.tile((12.0, -3.0), (len(run.swing_times), 1)),
        rtol=0.0,
        atol=0.0,
    )

    fitted = fit_run_torque_profile(
        run,
        profile_id="profile.rate.fitted.v1",
        name="Fitted Rate Miss",
        description="A reusable profile fitted from the retained miss history.",
        degree=0,
        source_metadata={"run_id": "run.miss.42"},
        created_at_utc="2026-08-05T12:00:00Z",
        modified_at_utc="2026-08-05T12:01:00Z",
    )
    assert fitted.source is TorqueProfileSource.FITTED_RUN
    assert fitted.model_id == DOUBLE_PENDULUM_MODEL_ID
    assert fitted.evaluate(0.025) == pytest.approx(
        {SHOULDER_JOINT_ID: 12.0, WRIST_JOINT_ID: -3.0}
    )
    assert all(
        assignment.polynomial.fit_metadata is not None
        for assignment in fitted.assignments
    )

    replay = _prescribed_miss(fitted)
    assert replay.impact_outcome.status is ImpactStatus.MISS
    np.testing.assert_allclose(replay.swing_poses, run.swing_poses, atol=1e-12)
    np.testing.assert_allclose(
        replay.swing_applied_torques_nm,
        run.swing_applied_torques_nm,
        atol=1e-12,
    )


def test_json_v5_and_csv_include_applied_torque_history(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    run = _prescribed_miss()
    payload = run_to_json_dict(run)
    torque_series = payload["series"]["swing_applied_joint_torques"]
    assert payload["format"] == "rate_of_closure.simulation_run/5"
    assert torque_series == {
        "unit": "N*m",
        "joint_ids": [SHOULDER_JOINT_ID, WRIST_JOINT_ID],
        "values": run.swing_applied_torques_nm.tolist(),
    }
    assert payload["impact_outcome"]["status"] == "miss"

    path = tmp_path / "torques.csv"
    write_torque_csv(run, path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(run.swing_times) * 2
    assert {
        row["applied_torque_nm"] for row in rows if row["joint_id"] == SHOULDER_JOINT_ID
    } == {"12.0"}
    assert {
        row["applied_torque_nm"] for row in rows if row["joint_id"] == WRIST_JOINT_ID
    } == {"-3.0"}


def test_rate_exports_shoulder_lock_and_retains_locked_miss_geometry() -> None:
    locks = JointLockConfig((SHOULDER_JOINT_ID,))
    run = run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=_DRIVER,
            source_kind="double_pendulum",
            swing_duration_s=_DURATION_S,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
            swing_run_config=DoublePendulumRunConfig(joint_locks=locks),
        )
    )
    assert run.impact_outcome.status is ImpactStatus.MISS
    np.testing.assert_allclose(
        run.swing_joints[:, 1],
        np.tile(run.swing_joints[0, 1], (len(run.swing_times), 1)),
        rtol=0.0,
        atol=0.0,
    )
    assert run_to_json_dict(run)["parameters"]["locked_joint_ids"] == [
        SHOULDER_JOINT_ID
    ]
    kinetics = compute_kinetics(
        run, analysis_time_s=run.impact_outcome.candidate_time_s
    )
    assert kinetics is not None
    np.testing.assert_array_equal(kinetics.torque_applied_nm, 0.0)
    assert np.abs(kinetics.torque_constraint_reaction_nm[:, 0]).max() > 0.0
    np.testing.assert_allclose(
        kinetics.torque_applied_nm + kinetics.torque_constraint_reaction_nm,
        kinetics.torque_inertial_nm
        - kinetics.torque_gravity_nm
        - kinetics.torque_damping_nm,
        rtol=0.0,
        atol=1e-12,
    )


def test_locked_kinetics_keep_commanded_and_constraint_torques_separate() -> None:
    profile = _prescribed_profile()
    run = run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=_DRIVER,
            source_kind="double_pendulum",
            swing_duration_s=_DURATION_S,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
            swing_run_config=DoublePendulumRunConfig.prescribed(
                profile.profile_id,
                joint_locks=JointLockConfig((SHOULDER_JOINT_ID,)),
            ),
            torque_library=TorqueProfileLibrary((profile,)),
        )
    )
    kinetics = compute_kinetics(
        run, analysis_time_s=run.impact_outcome.candidate_time_s
    )
    assert kinetics is not None
    np.testing.assert_array_equal(
        kinetics.torque_applied_nm, run.swing_applied_torques_nm
    )
    assert np.abs(kinetics.torque_constraint_reaction_nm[:, 0]).max() > 0.0


def test_rate_rejects_joint_locks_for_unsupported_triple_source() -> None:
    with pytest.raises(ContractViolationError, match="locks.*double-pendulum"):
        SimulationConfig(
            scenario=_SCENARIO,
            club=_DRIVER,
            source_kind="triple_pendulum",
            swing_run_config=DoublePendulumRunConfig(
                joint_locks=JointLockConfig((WRIST_JOINT_ID,))
            ),
        )


def test_source_factory_rejects_joint_locks_for_unsupported_triple_source() -> None:
    from rate_of_closure.simulation.sources import make_source

    with pytest.raises(ContractViolationError, match="locks.*unsupported"):
        make_source(
            "triple_pendulum",
            _SCENARIO,
            run_config=DoublePendulumRunConfig(
                joint_locks=JointLockConfig((WRIST_JOINT_ID,))
            ),
        )


@pytest.mark.parametrize("source_kind", ["manual", "triple_pendulum"])
def test_source_factory_rejects_localized_torque_for_unsupported_source(
    source_kind: str,
) -> None:
    from rate_of_closure.simulation.sources import make_source

    run_config = DoublePendulumRunConfig(
        commanded_torque_offsets=(
            LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.01, 0.02), 1.0),
        )
    )

    with pytest.raises(ContractViolationError, match="torque offsets.*unsupported"):
        make_source(source_kind, _SCENARIO, run_config=run_config)


@pytest.mark.parametrize(
    "run_config",
    [
        {},
        [],
        0,
        False,
        "",
        {"bad": 1},
        [1],
        pytest.param(object(), id="object"),
    ],
)
def test_source_factory_rejects_wrong_and_falsey_non_config_objects(
    run_config: object,
) -> None:
    from rate_of_closure.simulation.sources import make_source

    with pytest.raises(ContractViolationError, match="run_config"):
        make_source(  # type: ignore[arg-type]
            "manual", _SCENARIO, run_config=run_config
        )


@pytest.mark.parametrize("source_kind", ["manual", "triple_pendulum"])
@pytest.mark.parametrize(
    ("run_config", "message"),
    [
        (
            DoublePendulumRunConfig.prescribed("profile.test.v1"),
            "execution.*unsupported",
        ),
        (
            DoublePendulumRunConfig(joint_locks=JointLockConfig((WRIST_JOINT_ID,))),
            "locks.*unsupported",
        ),
        (
            DoublePendulumRunConfig(
                commanded_torque_offsets=(
                    LocalizedTorqueOffset(SHOULDER_JOINT_ID, (0.01, 0.02), 1.0),
                )
            ),
            "torque offsets.*unsupported",
        ),
    ],
)
def test_source_factory_rejects_all_nondefault_execution_for_unsupported_sources(
    source_kind: str,
    run_config: DoublePendulumRunConfig,
    message: str,
) -> None:
    from rate_of_closure.simulation.sources import make_source

    with pytest.raises(ContractViolationError, match=message):
        make_source(source_kind, _SCENARIO, run_config=run_config)


@pytest.mark.parametrize("source_kind", ["manual", "triple_pendulum"])
def test_source_factory_accepts_explicit_default_execution_for_every_source(
    source_kind: str,
) -> None:
    from rate_of_closure.simulation.sources import make_source

    source = make_source(
        source_kind,
        _SCENARIO,
        duration=0.05,
        run_config=DoublePendulumRunConfig(),
    )

    assert source.sample(0.0).pose.shape == (4, 4)
