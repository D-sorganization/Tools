"""Tests for integrating movement-optimizer as a SwingSource in Rate of Closure."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    SOURCE_KINDS,
    SimulationConfig,
    make_source,
    run_simulation,
)
from rate_of_closure.simulation.anthropometry import GolferAnthropometry
from shared.python.swing_sim.swing_source import SwingSource


def test_movement_optimizer_is_registered_in_source_kinds() -> None:
    assert "movement_optimizer" in SOURCE_KINDS


def test_make_source_movement_optimizer_with_anthropometry() -> None:
    scenario = ImpactScenario(clubhead_speed_mph=110.0)
    anthro = GolferAnthropometry(height_m=1.75, mass_kg=75.0)
    source = make_source(
        "movement_optimizer",
        scenario,
        golfer_anthropometry=anthro,
        duration=0.5,
    )
    assert isinstance(source, SwingSource)
    assert source.duration == pytest.approx(0.5)
    sample = source.sample(0.25)
    assert sample.pose.shape == (4, 4)
    assert sample.twist.shape == (6,)


def test_simulation_pipeline_with_movement_optimizer() -> None:
    scenario = ImpactScenario(clubhead_speed_mph=110.0)
    driver = get_club("Driver 10.5°")
    anthro = GolferAnthropometry(height_m=1.80, mass_kg=80.0)

    config = SimulationConfig(
        scenario=scenario,
        club=driver,
        source_kind="movement_optimizer",
        golfer_anthropometry=anthro,
        swing_duration_s=0.5,
    )

    run = run_simulation(config)
    assert run.config.source_kind == "movement_optimizer"
    assert len(run.swing_times) > 0
    assert run.swing_positions.shape[1] == 3


def test_simulation_pipeline_with_native_optimization_result() -> None:
    from movement_optimizer.trajectory.result import OptimizationResult

    scenario = ImpactScenario(clubhead_speed_mph=110.0)
    driver = get_club("Driver 10.5°")
    n = 20
    t = np.linspace(0.0, 0.5, n)
    poses = np.zeros((n, 4, 4), dtype=float)
    twists = np.zeros((n, 6), dtype=float)
    for i in range(n):
        poses[i] = np.eye(4)
        poses[i, 0, 3] = float(i) * 0.05
        twists[i] = np.array([0.0, 1.0, 0.0, 10.0, 0.0, 5.0])

    native_res = OptimizationResult(
        t=t,
        q=np.zeros((n, 3)),
        qd=np.zeros((n, 3)),
        qdd=np.zeros((n, 3)),
        torques=np.zeros((n, 3)),
        power=np.zeros((n, 3)),
        com=np.zeros((n, 2)),
        bar=np.zeros((n, 2)),
        success=True,
        cost=1.0,
        com_horizontal_range_cm=0.0,
        elapsed_s=0.1,
        n_evals=10,
        n_joint_limit_violations=0,
        clubhead_poses=poses,
        clubhead_twists=twists,
    )

    config = SimulationConfig(
        scenario=scenario,
        club=driver,
        source_kind="movement_optimizer",
        optimized_result=native_res,
        swing_duration_s=0.5,
    )
    assert config.optimized_result is native_res
    run = run_simulation(config)
    assert run.config.source_kind == "movement_optimizer"


def test_matches_golden_fixture_movement_optimizer_golden_v1() -> None:
    import json
    from pathlib import Path

    from rate_of_closure.simulation.sources import APP_FROM_SWING

    fixture_path = (
        Path(__file__).parents[2]
        / "src"
        / "rate_of_closure"
        / "web"
        / "src"
        / "model"
        / "__fixtures__"
        / "movement_optimizer_golden_v1.json"
    )
    with open(fixture_path) as f:
        golden = json.load(f)

    anthro_data = golden["golfer_anthropometry"]
    anthro = GolferAnthropometry(
        height_m=anthro_data["height_m"],
        mass_kg=anthro_data["mass_kg"],
    )
    res = anthro.generate_delivery(duration_s=golden["duration_s"], dt=golden["dt_s"])
    assert len(res.time_s) == len(golden["samples"])

    c = APP_FROM_SWING
    for i, exp in enumerate(golden["samples"]):
        t = res.time_s[i]
        assert t == pytest.approx(exp["t"], abs=1e-4)

        pos_app = c @ res.clubhead_poses[i][:3, 3]
        np.testing.assert_allclose(pos_app, exp["position"], atol=1e-3)

        v_app = c @ res.clubhead_twists[i][3:]
        np.testing.assert_allclose(v_app, exp["velocity"], atol=1e-3)

        w_app = c @ res.clubhead_twists[i][:3]
        np.testing.assert_allclose(w_app, exp["angular_velocity"], atol=1e-3)

        rot_app = c @ res.clubhead_poses[i][:3, :3]
        np.testing.assert_allclose(rot_app, exp["rotation"], atol=1e-3)

        joints_app = res.joint_positions_m[i] @ c.T
        np.testing.assert_allclose(joints_app, exp["joints"], atol=1e-3)


def test_lagrangian_kinematics_clubhead_pose_and_twist() -> None:
    from movement_optimizer.models.body_model import BodyModel
    from movement_optimizer.models.lagrangian_dynamics import LagrangianDynamics

    body = BodyModel(75.0, 1.75)
    dyn = LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), 0.0)
    q = np.array([0.1, 0.2, 0.3])
    qd = np.array([1.0, 2.0, 3.0])

    pose = dyn.clubhead_pose(q)
    assert pose.shape == (4, 4)
    assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)

    twist = dyn.clubhead_twist(q, qd)
    assert twist.shape == (6,)
    assert twist[1] == pytest.approx(3.0)
    assert np.linalg.norm(twist[3:]) > 0.1
