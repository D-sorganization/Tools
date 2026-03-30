import numpy as np
import pytest

from lower_body_model.builder import build_lower_body_xml
from lower_body_model.simulator import LowerBodySimulator


@pytest.fixture
def simulator() -> LowerBodySimulator:
    xml = build_lower_body_xml()
    return LowerBodySimulator(xml)


def test_initial_pelvis_kinematics(simulator: LowerBodySimulator) -> None:
    """Test that the pelvis starts at the expected height and zero tilts."""
    kinematics = simulator.compute_pelvis_kinematics()

    assert np.isclose(kinematics["x_forward"], 0.0, atol=1e-3)
    assert np.isclose(kinematics["y_lateral"], 0.0, atol=1e-3)
    assert kinematics["z_vertical"] > 0.8  # Around 0.9m + 0.1

    # Should have no tilts initially
    assert np.isclose(kinematics["roll"], 0.0, atol=1e-3)
    assert np.isclose(kinematics["pitch"], 0.0, atol=1e-3)
    assert np.isclose(kinematics["yaw"], 0.0, atol=1e-3)


def test_polynomial_driver(simulator: LowerBodySimulator) -> None:
    """Test that setting a polynomial driver changes the control signal over time."""
    simulator.set_joint_polynomial("act_r_hip_x", [2.0, 1.0])

    # Step the simulation by 1 second (dt=0.005, 200 steps)
    for _ in range(200):
        simulator.step()

    kinematics = simulator.compute_pelvis_kinematics()

    # Since we actuated the hip, the leg moved, pulling the body or falling
    assert (
        not np.isclose(kinematics["x_forward"], 0.0, atol=1e-2)
        or not np.isclose(kinematics["y_lateral"], 0.0, atol=1e-2)
        or not np.isclose(kinematics["z_vertical"], 1.0, atol=1e-2)
    )


def test_zero_torque_counterfactual(simulator: LowerBodySimulator) -> None:
    """Test the counterfactual forces computation."""
    ztcf_initial = simulator.compute_zero_torque_counterfactual()

    assert "qacc_ztcf" in ztcf_initial
    assert "qfrc_to_hold_state" in ztcf_initial

    # Drop for a few steps so it gets velocity
    for _ in range(10):
        simulator.step()

    ztcf_falling = simulator.compute_zero_torque_counterfactual()

    diff = np.linalg.norm(
        ztcf_initial["qfrc_to_hold_state"] - ztcf_falling["qfrc_to_hold_state"]
    )
    assert diff > 0.0, "Forces to hold state should change as the model falls."


def test_induced_acceleration_analysis(simulator: LowerBodySimulator) -> None:
    """Test induced acceleration by an actuator."""
    # Compute IAA for right hip x
    iaa_result = simulator.analyze_induced_acceleration(
        "act_r_hip_x", torque_value=10.0
    )

    # Driving the right hip in x (which is the forward/backward axis or lateral depending on orientation)
    # Should induce some acceleration on the floating pelvis.
    assert "forward_accel" in iaa_result
    assert "lateral_accel" in iaa_result
    assert "yaw_accel" in iaa_result

    # The total induced acceleration shouldn't be identically perfectly zero
    total_accel = sum(abs(v) for v in iaa_result.values())
    assert total_accel > 1e-4, (
        "Applied torque should induce some acceleration on the root body."
    )
