"""Contracts for the profile library and fit-from-run export boundary."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.run_config import (
    DOUBLE_PENDULUM_MODEL_ID,
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
)
from shared.python.swing_sim.torque_export import fit_torque_history_profile
from shared.python.swing_sim.torque_library import TorqueProfileLibrary
from shared.python.swing_sim.torque_profiles import (
    JointTorqueAssignment,
    PrescribedTorqueProfile,
    TorquePolynomial,
    TorqueProfileSource,
)

pytestmark = pytest.mark.unit


def _profile(profile_id: str = "profile.one") -> PrescribedTorqueProfile:
    return PrescribedTorqueProfile(
        profile_id=profile_id,
        model_id=DOUBLE_PENDULUM_MODEL_ID,
        name="Profile One",
        description="A profile for library contract tests.",
        source=TorqueProfileSource.DIRECT,
        source_metadata={"author": "test-suite"},
        created_at_utc="2026-08-05T12:00:00Z",
        modified_at_utc="2026-08-05T12:00:00Z",
        time_domain_s=(0.0, 1.0),
        assignments=(
            JointTorqueAssignment(SHOULDER_JOINT_ID, TorquePolynomial((1.0, 2.0))),
            JointTorqueAssignment(WRIST_JOINT_ID, TorquePolynomial((-1.0,))),
        ),
    )


class TestTorqueProfileLibrary:
    def test_resolves_profiles_by_stable_id(self) -> None:
        profile = _profile()
        library = TorqueProfileLibrary((profile,))
        assert library.get(profile.profile_id) is profile
        assert library.for_model(DOUBLE_PENDULUM_MODEL_ID) == (profile,)

    def test_empty_library_is_valid_but_missing_lookup_is_actionable(self) -> None:
        library = TorqueProfileLibrary()
        with pytest.raises(ContractViolationError, match="not found"):
            library.get("profile.missing")

    def test_rejects_duplicate_profile_ids(self) -> None:
        profile = _profile()
        with pytest.raises(ContractViolationError, match="unique"):
            TorqueProfileLibrary((profile, profile))

    def test_with_profile_returns_a_new_library(self) -> None:
        original = TorqueProfileLibrary()
        updated = original.with_profile(_profile())
        assert original.profiles == ()
        assert updated.get("profile.one").name == "Profile One"


class TestFitTorqueHistoryProfile:
    def test_exports_each_joint_as_a_reusable_fitted_profile(self) -> None:
        times_s = np.linspace(0.0, 1.0, 21)
        histories = {
            WRIST_JOINT_ID: -2.0 + 0.5 * times_s,
            SHOULDER_JOINT_ID: 4.0 - 3.0 * times_s + times_s**2,
        }
        profile = fit_torque_history_profile(
            profile_id="profile.fitted_run.42",
            model_id=DOUBLE_PENDULUM_MODEL_ID,
            name="Fitted Run 42",
            description="Polynomial reconstruction of sampled run torques.",
            source_metadata={"run_id": "run.42"},
            created_at_utc="2026-08-05T12:00:00Z",
            modified_at_utc="2026-08-05T12:01:00Z",
            times_s=times_s,
            torque_nm_by_joint=histories,
            degree=2,
        )

        assert profile.source is TorqueProfileSource.FITTED_RUN
        assert profile.time_domain_s == (0.0, 1.0)
        assert tuple(item.joint_id for item in profile.assignments) == (
            SHOULDER_JOINT_ID,
            WRIST_JOINT_ID,
        )
        assert profile.evaluate(0.4) == pytest.approx(
            {
                SHOULDER_JOINT_ID: 4.0 - 3.0 * 0.4 + 0.4**2,
                WRIST_JOINT_ID: -2.0 + 0.5 * 0.4,
            }
        )
        for assignment in profile.assignments:
            metadata = assignment.polynomial.fit_metadata
            assert metadata is not None
            assert metadata.degree == 2
            assert metadata.r_squared == pytest.approx(1.0)
            assert metadata.original_sample_sha256 is not None
        assert TorqueProfileLibrary((profile,)).get(profile.profile_id) == profile

    @pytest.mark.parametrize(
        "histories",
        [
            {},
            {"bad joint": [1.0, 2.0, 3.0]},
            {SHOULDER_JOINT_ID: [1.0, 2.0]},
        ],
    )
    def test_rejects_empty_invalid_or_mismatched_histories(
        self, histories: dict[str, list[float]]
    ) -> None:
        with pytest.raises(ContractViolationError):
            fit_torque_history_profile(
                profile_id="profile.invalid",
                model_id=DOUBLE_PENDULUM_MODEL_ID,
                name="Invalid",
                description="Invalid history fixture.",
                source_metadata={"run_id": "run.invalid"},
                created_at_utc="2026-08-05T12:00:00Z",
                modified_at_utc="2026-08-05T12:00:00Z",
                times_s=[0.0, 0.5, 1.0],
                torque_nm_by_joint=histories,
                degree=1,
            )

    def test_rejects_fit_above_requested_condition_limit(self) -> None:
        with pytest.raises(ContractViolationError, match="condition"):
            fit_torque_history_profile(
                profile_id="profile.ill_conditioned",
                model_id=DOUBLE_PENDULUM_MODEL_ID,
                name="Ill Conditioned",
                description="Fit intentionally rejected by a strict limit.",
                source_metadata={"run_id": "run.ill"},
                created_at_utc="2026-08-05T12:00:00Z",
                modified_at_utc="2026-08-05T12:00:00Z",
                times_s=[0.0, 0.5, 1.0],
                torque_nm_by_joint={SHOULDER_JOINT_ID: [0.0, 1.0, 0.0]},
                degree=2,
                max_condition_number=1.0,
            )
