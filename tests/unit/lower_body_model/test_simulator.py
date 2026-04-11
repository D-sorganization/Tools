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
    assert (
        total_accel > 1e-4
    ), "Applied torque should induce some acceleration on the root body."


def test_history_recording_and_restoring(simulator: LowerBodySimulator) -> None:
    """Test that simulator tracks history and can restore previous frames for scrubbing."""
    assert len(simulator.history) == 0

    simulator.step()
    assert len(simulator.history) == 1

    # Run a few steps
    for _ in range(5):
        simulator.step()

    assert len(simulator.history) == 6

    # Store current state
    current_time = simulator.data.time

    # Restore an old frame
    simulator.restore_frame(0)
    assert simulator.data.time < current_time
    assert simulator.data.time == simulator.history[0]["time"]

    simulator.clear_history()
    assert len(simulator.history) == 0


def test_stability_properties(simulator: LowerBodySimulator) -> None:
    """Test that setting pd controls updates the variables correctly."""
    simulator.kp_stability = 250.0
    simulator.kd_stability = 25.0

    assert simulator.kp_stability == 250.0
    assert simulator.kd_stability == 25.0


def test_compute_diagnostics(simulator: LowerBodySimulator) -> None:
    """Test that diagnostics are computed fully without crashing."""
    simulator.setup_initial_pose(
        hip_anterior_tilt=10.0, knee_flexion=20.0, foot_angle=5.0
    )

    diag = simulator.compute_diagnostics()
    assert "time_sec" in diag
    assert "pelvis_z_m" in diag
    assert "is_diverged" in diag
    assert "max_tracking_err_deg" in diag
    assert "total_applied_torque_nm" in diag
    assert "r_knee_deg" in diag
    assert not diag["is_diverged"]

    # ensure it updates during playback
    for _ in range(50):
        simulator.step()

    diag_later = simulator.compute_diagnostics()
    assert float(diag_later["time_sec"]) > float(diag["time_sec"])
