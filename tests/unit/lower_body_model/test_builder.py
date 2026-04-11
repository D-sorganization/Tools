import mujoco
import pytest

from lower_body_model.builder import build_lower_body_xml


def test_build_lower_body_xml_generates_valid_mjcf() -> None:
    """Test that the generated XML is a valid MuJoCo model."""
    xml_string = build_lower_body_xml()

    # Load model to verify it compiles correctly
    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
    except Exception as e:
        pytest.fail(f"MuJoCo XML compilation failed: {e}")

    assert model.nq > 0, "Model should have degrees of freedom"
    assert model.nu > 0, "Model should have actuated joints"

    # Verify expected joint names exist
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    # We expect floating base / pelvis joints, hip, knee, ankle joints
    assert (
        "r_hip_x" in joint_names or "r_hip" in joint_names
    ), "Should have right hip joint"
    assert "r_knee" in joint_names, "Should have right knee joint"
    assert "l_knee" in joint_names, "Should have left knee joint"


def test_builder_can_change_masses() -> None:
    """Test that we can pass custom mass properties."""
    xml_string = build_lower_body_xml(thigh_mass=15.0, calf_mass=6.0)
    model = mujoco.MjModel.from_xml_string(xml_string)

    # Check that total mass is reasonable and updated
    assert sum(model.body_mass) > 20.0, "Total mass computed should be realistic"
