"""Regression coverage for prescribed-run kinetics reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    ContactMode,
    SimulationConfig,
    compute_kinetics,
    run_simulation,
)
from shared.python.swing_sim.run_config import (
    DOUBLE_PENDULUM_MODEL_ID,
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
    DoublePendulumRunConfig,
)
from shared.python.swing_sim.torque_library import TorqueProfileLibrary
from shared.python.swing_sim.torque_profiles import (
    JointTorqueAssignment,
    PrescribedTorqueProfile,
    TorquePolynomial,
    TorqueProfileSource,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_DURATION_S = 0.10
_SCENARIO = ImpactScenario(clubhead_speed_mph=113.0)
_DRIVER = get_club("Driver 10.5°")


def _profile() -> PrescribedTorqueProfile:
    return PrescribedTorqueProfile(
        profile_id="profile.kinetics.regression.v1",
        model_id=DOUBLE_PENDULUM_MODEL_ID,
        name="Kinetics Regression",
        description="Nonzero torque profile for prescribed kinetics reconstruction.",
        source=TorqueProfileSource.DIRECT,
        source_metadata={"author": "rate-test-suite"},
        created_at_utc="2026-08-05T12:00:00Z",
        modified_at_utc="2026-08-05T12:00:00Z",
        time_domain_s=(0.0, _DURATION_S),
        assignments=(
            JointTorqueAssignment(SHOULDER_JOINT_ID, TorquePolynomial((20.0,))),
            JointTorqueAssignment(WRIST_JOINT_ID, TorquePolynomial((-8.0,))),
        ),
    )


def _run(prescribed: bool):  # type: ignore[no-untyped-def]
    profile = _profile()
    return run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=_DRIVER,
            source_kind="double_pendulum",
            swing_duration_s=_DURATION_S,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
            swing_run_config=(
                DoublePendulumRunConfig.prescribed(profile.profile_id)
                if prescribed
                else DoublePendulumRunConfig()
            ),
            torque_library=(TorqueProfileLibrary((profile,)) if prescribed else None),
        )
    )


def test_prescribed_kinetics_reconstruct_the_prescribed_states() -> None:
    prescribed = _run(True)
    passive = _run(False)
    prescribed_series = compute_kinetics(
        prescribed,
        analysis_time_s=prescribed.impact_outcome.candidate_time_s,
    )
    passive_series = compute_kinetics(
        passive,
        analysis_time_s=passive.impact_outcome.candidate_time_s,
    )

    assert prescribed_series is not None and passive_series is not None
    assert not np.allclose(
        prescribed_series.torque_inertial_nm,
        passive_series.torque_inertial_nm,
        atol=1e-6,
    )
    # Inverse dynamics differentiates the sampled rates. Excluding two samples
    # at each boundary keeps this on the O(dt²) central-difference stencil; the
    # 0.02 N·m tolerance covers RK4 and differencing error at dt = 1 ms while
    # remaining three orders below the imposed shoulder torque.
    np.testing.assert_allclose(
        prescribed_series.torque_applied_nm[2:-2],
        prescribed.swing_applied_torques_nm[2:-2],
        rtol=0.0,
        atol=0.02,
    )
    np.testing.assert_allclose(
        prescribed_series.wrist_positions_m,
        prescribed.swing_joints[:, 1, :],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        prescribed_series.clubhead_positions_m,
        prescribed.swing_positions,
        atol=1e-12,
    )
