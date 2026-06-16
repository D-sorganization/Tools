from __future__ import annotations

import math

import pytest

from lower_body_model.builder import build_lower_body_xml

mujoco = pytest.importorskip("mujoco")

from lower_body_model.simulator import LowerBodySimulator


def test_builder_rejects_non_positive_geometry() -> None:
    with pytest.raises(ValueError, match="thigh_length must be strictly positive"):
        build_lower_body_xml(thigh_length=0.0)


def test_simulator_constructs_bilateral_joint_and_actuator_contract() -> None:
    simulator = LowerBodySimulator(build_lower_body_xml())

    expected_joints = {
        "root",
        "r_hip_x",
        "r_hip_y",
        "r_hip_z",
        "r_knee",
        "r_ankle_x",
        "r_ankle_y",
        "l_hip_x",
        "l_hip_y",
        "l_hip_z",
        "l_knee",
        "l_ankle_x",
        "l_ankle_y",
    }
    expected_actuators = {
        f"act_{joint}" for joint in expected_joints if joint != "root"
    }

    assert expected_joints.issubset(set(simulator.joint_names))
    assert expected_actuators.issubset(set(simulator.actuator_names))
    assert simulator.model.njnt >= len(expected_joints)
    assert simulator.model.nu == len(expected_actuators)


def test_initial_pose_sets_finite_stability_target() -> None:
    simulator = LowerBodySimulator(build_lower_body_xml())

    simulator.setup_initial_pose(
        hip_anterior_tilt=10.0,
        knee_flexion=20.0,
        foot_angle=15.0,
    )

    assert simulator.qpos_target is not None
    assert simulator.qpos_target.shape == simulator.data.qpos.shape
    assert all(math.isfinite(float(value)) for value in simulator.qpos_target)
    assert simulator.data.qpos[simulator.jnt_qpos_idx["r_knee"]] == pytest.approx(
        math.radians(20.0)
    )
    assert simulator.data.qpos[simulator.jnt_qpos_idx["l_knee"]] == pytest.approx(
        math.radians(20.0)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hip_anterior_tilt": 91.0}, "hip_anterior_tilt must be in"),
        ({"knee_flexion": -1.0}, "knee_flexion must be in"),
        ({"foot_angle": 91.0}, "foot_angle must be in"),
    ],
)
def test_initial_pose_rejects_out_of_range_inputs(
    kwargs: dict[str, float],
    message: str,
) -> None:
    simulator = LowerBodySimulator(build_lower_body_xml())

    with pytest.raises(ValueError, match=message):
        simulator.setup_initial_pose(**kwargs)
