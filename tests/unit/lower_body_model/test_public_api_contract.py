"""Contract tests for the lower_body_model public API.

These tests lock down the surface that downstream repos (UpstreamDrift,
Gasification_Model) and in-repo callers rely on. Breaking any of them is
a signal that a public API change slipped through without a deprecation
path. They are kept deliberately shallow — we check signatures, return
keys, exception types, and existence rather than deep behaviour, which
is covered by the regular unit tests.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from lower_body_model import HipRotationSample, InclinedPlaneHipRotationTarget
from lower_body_model.builder import build_lower_body_xml
from lower_body_model.hip_rotation import (
    HipRotationSample as DirectSample,
)
from lower_body_model.hip_rotation import (
    InclinedPlaneHipRotationTarget as DirectTarget,
)
from lower_body_model.simulator import LowerBodySimulator

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# Package-level exports
# ---------------------------------------------------------------------------


def test_package_exports_hip_rotation_types() -> None:
    assert HipRotationSample is DirectSample
    assert InclinedPlaneHipRotationTarget is DirectTarget


# ---------------------------------------------------------------------------
# build_lower_body_xml signature
# ---------------------------------------------------------------------------


def test_build_lower_body_xml_signature() -> None:
    sig = inspect.signature(build_lower_body_xml)
    expected = {
        "thigh_mass",
        "calf_mass",
        "foot_mass",
        "pelvis_mass",
        "thigh_length",
        "calf_length",
        "pelvis_width",
    }
    assert expected.issubset(sig.parameters.keys())
    for name in expected:
        assert sig.parameters[name].default is not inspect.Parameter.empty


def test_build_lower_body_xml_returns_mjcf_str() -> None:
    xml = build_lower_body_xml()
    assert isinstance(xml, str)
    assert xml.startswith("<mujoco")


# ---------------------------------------------------------------------------
# InclinedPlaneHipRotationTarget contract
# ---------------------------------------------------------------------------


def test_target_public_fields_and_defaults() -> None:
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0)
    for field in (
        "duration_sec",
        "backswing_degrees",
        "counterclockwise_degrees",
        "incline_degrees",
        "sample_count",
        "lateral_shift_m",
    ):
        assert hasattr(target, field), f"missing field: {field}"


def test_target_public_methods_present() -> None:
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0)
    for method_name in (
        "rotation_degrees_at",
        "plane_point_at",
        "lateral_shift_at",
        "target_quaternion_at",
        "sample",
    ):
        assert callable(getattr(target, method_name)), f"missing: {method_name}"


def test_target_sample_returns_hip_rotation_samples() -> None:
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0, sample_count=3)
    samples = target.sample()
    assert len(samples) == 3
    assert all(isinstance(s, HipRotationSample) for s in samples)
    for s in samples:
        assert hasattr(s, "time_sec")
        assert hasattr(s, "rotation_deg")
        assert hasattr(s, "plane_point")


# ---------------------------------------------------------------------------
# LowerBodySimulator surface
# ---------------------------------------------------------------------------


@pytest.fixture
def simulator() -> LowerBodySimulator:
    return LowerBodySimulator(build_lower_body_xml())


def test_simulator_public_methods_present(simulator: LowerBodySimulator) -> None:
    """Locked-down public method surface.

    Do not remove or rename any of these without a deprecation shim;
    downstream repos and the PyQt6 control panel call into them by name.
    """
    required = (
        "setup_initial_pose",
        "reset",
        "set_joint_polynomial",
        "configure_hip_rotation_target",
        "apply_hip_rotation_target",
        "set_pelvis_inclined_rotation",
        "clear_pelvis_inclined_rotation",
        "compute_zero_torque_counterfactual",
        "compute_pelvis_kinematics",
        "compute_diagnostics",
        "analyze_induced_acceleration",
        "inverse_kinematics",
        "step",
        "restore_frame",
        "get_history_diagnostics",
        "clear_history",
    )
    for name in required:
        assert callable(getattr(simulator, name, None)), f"missing: {name}"


def test_compute_pelvis_kinematics_return_shape(
    simulator: LowerBodySimulator,
) -> None:
    result = simulator.compute_pelvis_kinematics()
    for key in ("x_forward", "y_lateral", "z_vertical", "roll", "pitch", "yaw"):
        assert key in result
        assert isinstance(result[key], float)


def test_compute_diagnostics_return_shape(simulator: LowerBodySimulator) -> None:
    simulator.setup_initial_pose()
    diag = simulator.compute_diagnostics()
    for key in (
        "time_sec",
        "pelvis_z_m",
        "is_diverged",
        "max_tracking_err_deg",
        "total_applied_torque_nm",
        "r_knee_deg",
        "history_frames",
        "grf",
        "joint_torques",
    ):
        assert key in diag, f"diagnostics missing {key}"
    assert isinstance(diag["grf"], dict)
    assert {"right_z", "left_z"}.issubset(diag["grf"].keys())


def test_analyze_induced_acceleration_return_shape(
    simulator: LowerBodySimulator,
) -> None:
    iaa = simulator.analyze_induced_acceleration("act_r_hip_x", 1.0)
    for key in (
        "forward_accel",
        "lateral_accel",
        "vertical_accel",
        "roll_accel",
        "pitch_accel",
        "yaw_accel",
    ):
        assert key in iaa


def test_setup_initial_pose_raises_value_error_type(
    simulator: LowerBodySimulator,
) -> None:
    """DbC: out-of-range inputs must raise ValueError exactly."""
    with pytest.raises(ValueError):
        simulator.setup_initial_pose(hip_anterior_tilt=999.0)


def test_inverse_kinematics_return_type(simulator: LowerBodySimulator) -> None:
    simulator.setup_initial_pose()
    pos = simulator.data.qpos[0:3].copy()
    quat = simulator.data.qpos[3:7].copy()
    result = simulator.inverse_kinematics(pos, quat, max_iters=5)
    assert isinstance(result, bool)


def test_set_pelvis_inclined_rotation_uses_target_type(
    simulator: LowerBodySimulator,
) -> None:
    target = InclinedPlaneHipRotationTarget(
        duration_sec=1.0, lateral_shift_m=0.05, incline_degrees=10.0
    )
    simulator.set_pelvis_inclined_rotation(target)
    # Stepping should be a no-op in terms of API (returns None).
    assert simulator.step() is None


def test_analyze_induced_acceleration_raises_value_error_on_missing_actuator(
    simulator: LowerBodySimulator,
) -> None:
    with pytest.raises(ValueError):
        simulator.analyze_induced_acceleration("act_does_not_exist", 1.0)


def test_target_quaternion_at_returns_numpy_quaternion() -> None:
    target = InclinedPlaneHipRotationTarget(duration_sec=1.0)
    q = target.target_quaternion_at(0.5)
    assert isinstance(q, np.ndarray)
    assert q.shape == (4,)
    # Unit norm.
    assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=1e-6)
