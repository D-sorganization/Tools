"""Contracts for explicit prescribed-torque double-pendulum runs."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.run_config import (
    DOUBLE_PENDULUM_MODEL_ID,
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
    DoublePendulumRunConfig,
    SwingRunMode,
)
from shared.python.swing_sim.swing_source import DoublePendulumSwing
from shared.python.swing_sim.torque_library import TorqueProfileLibrary
from shared.python.swing_sim.torque_profiles import (
    JointTorqueAssignment,
    PrescribedTorqueProfile,
    TorquePolynomial,
    TorqueProfileSource,
)
from shared.python.swing_sim.types import PendulumState

pytestmark = pytest.mark.unit


def _profile(
    *,
    profile_id: str = "profile.constant_drive.v1",
    model_id: str = DOUBLE_PENDULUM_MODEL_ID,
    joint_ids: tuple[str, str] = (SHOULDER_JOINT_ID, WRIST_JOINT_ID),
    time_domain_s: tuple[float, float] = (0.0, 0.1),
) -> PrescribedTorqueProfile:
    return PrescribedTorqueProfile(
        profile_id=profile_id,
        model_id=model_id,
        name="Constant Drive",
        description="Constant shoulder and wrist torques for integration tests.",
        source=TorqueProfileSource.DIRECT,
        source_metadata={"author": "test-suite"},
        created_at_utc="2026-08-05T12:00:00Z",
        modified_at_utc="2026-08-05T12:00:00Z",
        time_domain_s=time_domain_s,
        assignments=(
            JointTorqueAssignment(joint_ids[0], TorquePolynomial((20.0,))),
            JointTorqueAssignment(joint_ids[1], TorquePolynomial((-5.0,))),
        ),
    )


def _prescribed_swing(
    profile: PrescribedTorqueProfile | None = None,
    **kwargs: object,
) -> DoublePendulumSwing:
    selected = profile or _profile()
    defaults: dict[str, object] = {
        "duration": 0.1,
        "dt": 0.001,
        "backend": "python",
        "gravity_m_s2": 0.0,
        "initial_state": PendulumState(0.0, 0.0, 0.0, 0.0),
        "run_config": DoublePendulumRunConfig.prescribed(selected.profile_id),
        "torque_library": TorqueProfileLibrary((selected,)),
    }
    defaults.update(kwargs)
    return DoublePendulumSwing(**defaults)  # type: ignore[arg-type]


class TestRunConfig:
    def test_run_mode_vocabulary_and_passive_default(self) -> None:
        assert {mode.value for mode in SwingRunMode} == {"passive", "prescribed"}
        config = DoublePendulumRunConfig()
        assert config.mode is SwingRunMode.PASSIVE
        assert config.prescribed_profile_id is None

    def test_prescribed_factory_requires_stable_profile_id(self) -> None:
        config = DoublePendulumRunConfig.prescribed("profile.driver.v1")
        assert config.mode is SwingRunMode.PRESCRIBED
        assert config.prescribed_profile_id == "profile.driver.v1"
        with pytest.raises(ContractViolationError, match="profile_id"):
            DoublePendulumRunConfig.prescribed("contains spaces")

    @pytest.mark.parametrize(
        ("mode", "profile_id"),
        [
            (SwingRunMode.PASSIVE, "profile.invalid"),
            (SwingRunMode.PRESCRIBED, None),
        ],
    )
    def test_rejects_mode_profile_mismatch(
        self, mode: SwingRunMode, profile_id: str | None
    ) -> None:
        with pytest.raises(ContractViolationError, match="profile"):
            DoublePendulumRunConfig(mode=mode, prescribed_profile_id=profile_id)

    def test_rejects_untyped_mode(self) -> None:
        with pytest.raises(ContractViolationError, match="run mode"):
            DoublePendulumRunConfig(mode="prescribed")  # type: ignore[arg-type]


class TestPrescribedSwing:
    def test_passive_default_keeps_existing_integration_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_forced(*_args: object, **_kwargs: object) -> np.ndarray:
            raise AssertionError("forced integration was selected")

        monkeypatch.setattr(
            "shared.python.swing_sim.reference.simulate_forced", fail_forced
        )
        swing = DoublePendulumSwing(duration=0.01, dt=0.001, backend="python")
        assert swing.run_mode is SwingRunMode.PASSIVE

    def test_prescribed_torques_drive_the_existing_dynamics(self) -> None:
        forced = _prescribed_swing()
        passive = DoublePendulumSwing(
            duration=0.1,
            dt=0.001,
            backend="python",
            gravity_m_s2=0.0,
            initial_state=PendulumState(0.0, 0.0, 0.0, 0.0),
        )

        assert forced.run_mode is SwingRunMode.PRESCRIBED
        assert forced.backend == "python"
        assert forced.joint_torques_at(0.05) == pytest.approx(
            {SHOULDER_JOINT_ID: 20.0, WRIST_JOINT_ID: -5.0}
        )
        forced_state = forced.state_at(forced.duration)
        passive_state = passive.state_at(passive.duration)
        assert abs(forced_state.omega1) + abs(forced_state.omega2) > 0.0
        assert passive_state == PendulumState(0.0, 0.0, 0.0, 0.0)

    def test_passive_torque_query_returns_stable_zero_assignments(self) -> None:
        swing = DoublePendulumSwing(duration=0.01, dt=0.001, backend="python")
        assert swing.joint_torques_at(0.005) == {
            SHOULDER_JOINT_ID: 0.0,
            WRIST_JOINT_ID: 0.0,
        }

    def test_auto_prescribed_mode_uses_explicit_python_forced_path(self) -> None:
        assert _prescribed_swing(backend="auto").backend == "python"

    def test_prescribed_replay_is_deterministic(self) -> None:
        first = _prescribed_swing()
        second = _prescribed_swing()
        for time_s in np.linspace(0.0, first.duration, 11):
            assert first.state_at(float(time_s)) == second.state_at(float(time_s))

    def test_rust_prescribed_mode_fails_instead_of_silently_ignoring_torque(
        self,
    ) -> None:
        with pytest.raises(ContractViolationError, match="prescribed.*Rust"):
            _prescribed_swing(backend="rust")

    @pytest.mark.parametrize(
        ("profile", "profile_id", "message"),
        [
            (_profile(model_id="model.other.v1"), None, "model_id"),
            (
                _profile(joint_ids=(SHOULDER_JOINT_ID, "joint.unknown")),
                None,
                "joint",
            ),
            (_profile(time_domain_s=(0.01, 0.1)), None, "time domain"),
            (_profile(), "profile.missing", "not found"),
        ],
    )
    def test_rejects_incompatible_or_missing_profile(
        self,
        profile: PrescribedTorqueProfile,
        profile_id: str | None,
        message: str,
    ) -> None:
        selected_id = profile_id or profile.profile_id
        with pytest.raises(ContractViolationError, match=message):
            _prescribed_swing(
                profile,
                run_config=DoublePendulumRunConfig.prescribed(selected_id),
            )

    def test_prescribed_mode_requires_a_library(self) -> None:
        profile = _profile()
        with pytest.raises(ContractViolationError, match="library"):
            DoublePendulumSwing(
                duration=0.1,
                dt=0.001,
                backend="python",
                run_config=DoublePendulumRunConfig.prescribed(profile.profile_id),
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"backend": "invalid"},
            {"run_config": "prescribed"},
            {"torque_library": "library"},
        ],
    )
    def test_rejects_untyped_execution_configuration(
        self, kwargs: dict[str, object]
    ) -> None:
        with pytest.raises(ContractViolationError):
            DoublePendulumSwing(duration=0.01, dt=0.001, **kwargs)  # type: ignore[arg-type]
