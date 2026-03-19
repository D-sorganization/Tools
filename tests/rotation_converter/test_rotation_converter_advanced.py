from __future__ import annotations

import math

import numpy as np
import pytest

from rotation_converter.advanced_kinematics import DualQuaternion, dh_to_matrix, slerp
from rotation_converter.converter import Rotation
from rotation_converter.core import (
    axis_angle_to_quaternion,
    axis_angle_to_rotation_matrix,
    euler_to_quaternion,
    normalize_quaternion,
    quaternion_to_euler,
)


def test_dh_standard() -> None:
    # Frame 0 to Frame 1 with 90 deg rotation about Z and unit translation along X
    T = dh_to_matrix(theta=math.pi / 2, d=0.0, a=1.0, alpha=0.0, modified=False)

    # Expected: [-y, x, z, [0, 1, 0]]
    expected_R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected_p = np.array([0.0, 1.0, 0.0])

    np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-10)
    np.testing.assert_allclose(T[:3, 3], expected_p, atol=1e-10)
    assert T.shape == (4, 4)


def test_slerp() -> None:
    # 90 degrees about Z
    q1 = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    q2 = normalize_quaternion(
        np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    )

    q_mid = slerp(q1, q2, 0.5)
    expected_mid = normalize_quaternion(
        np.array([math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)])
    )
    np.testing.assert_allclose(q_mid, expected_mid, atol=1e-10)


def test_dual_quaternion_basic() -> None:
    t = np.array([1.0, 2.0, 3.0])
    r = np.array([1.0, 0.0, 0.0, 0.0])  # Unit w=1

    dq = DualQuaternion.from_translation_rotation(translation=t, rotation_quaternion=r)

    # Translation should be perfectly extracted
    np.testing.assert_allclose(dq.extract_translation(), t, atol=1e-10)
    np.testing.assert_allclose(dq.real, r, atol=1e-10)


def test_dual_quaternion_multiply() -> None:
    # Frame 1 offset from Frame 0 by X=1
    dq1 = DualQuaternion.from_translation_rotation([1.0, 0, 0], [1.0, 0, 0, 0])

    # Frame 2 offset from Frame 1 by Y=1
    dq2 = DualQuaternion.from_translation_rotation([0, 1.0, 0], [1.0, 0, 0, 0])

    dq_composed = dq1.multiply(dq2)
    new_t = dq_composed.extract_translation()

    np.testing.assert_allclose(new_t, [1.0, 1.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# DbC precondition tests — GH1483
# Verify that bare assert has been replaced by explicit raise (TypeError),
# which is NOT disabled in Python optimized mode (-O).
# ---------------------------------------------------------------------------


def test_dh_to_matrix_rejects_none_theta() -> None:
    """dh_to_matrix raises TypeError for non-numeric theta (DbC fix)."""
    with pytest.raises(TypeError):
        dh_to_matrix(theta=None, d=0.0, a=1.0, alpha=0.0)  # type: ignore[arg-type]


def test_slerp_rejects_none_q1() -> None:
    """slerp raises TypeError for None q1 (DbC fix)."""
    q2 = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        slerp(None, q2, 0.5)  # type: ignore[arg-type]


def test_slerp_rejects_none_q2() -> None:
    """slerp raises TypeError for None q2 (DbC fix)."""
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        slerp(q1, None, 0.5)  # type: ignore[arg-type]


def test_dual_quaternion_multiply_rejects_non_dq() -> None:
    """DualQuaternion.multiply raises TypeError for non-DualQuaternion (DbC fix)."""
    dq = DualQuaternion.from_translation_rotation([1.0, 0, 0], [1.0, 0, 0, 0])
    with pytest.raises(TypeError):
        dq.multiply("not a DualQuaternion")  # type: ignore[arg-type]


def test_rotation_init_rejects_none() -> None:
    """Rotation.__init__ raises TypeError for None (DbC fix)."""
    with pytest.raises(TypeError):
        Rotation(None)  # type: ignore[arg-type]


def test_rotation_from_euler_rejects_non_numeric_a() -> None:
    """Rotation.from_euler raises TypeError for non-numeric angle a (DbC fix)."""
    with pytest.raises(TypeError):
        Rotation.from_euler(None, 0.0, 0.0, "xyz")  # type: ignore[arg-type]


def test_rotation_from_axis_angle_rejects_non_numeric_angle() -> None:
    """Rotation.from_axis_angle raises TypeError for non-numeric angle (DbC fix)."""
    axis = np.array([0.0, 0.0, 1.0])
    with pytest.raises(TypeError):
        Rotation.from_axis_angle(axis, None)  # type: ignore[arg-type]


def test_rotation_compose_rejects_non_rotation() -> None:
    """Rotation.compose raises TypeError for non-Rotation argument (DbC fix)."""
    r = Rotation.identity()
    with pytest.raises(TypeError):
        r.compose("not a rotation")  # type: ignore[arg-type]


def test_rotation_compose_lod_fix_correct_result() -> None:
    """Rotation.compose produces correct result after LoD fix (uses as_quaternion())."""
    r1 = Rotation.from_euler(math.pi / 2, 0.0, 0.0, "xyz")
    r2 = Rotation.from_euler(0.0, math.pi / 2, 0.0, "xyz")
    composed = r1.compose(r2)
    # Round-trip: compose then decompose should match direct multiplication
    q1 = r1.as_quaternion()
    q2 = r2.as_quaternion()
    from rotation_converter.core import quaternion_multiply

    expected_q = normalize_quaternion(quaternion_multiply(q1, q2))
    np.testing.assert_allclose(composed.as_quaternion(), expected_q, atol=1e-10)


def test_axis_angle_to_quaternion_rejects_non_numeric_angle() -> None:
    """axis_angle_to_quaternion raises TypeError for non-numeric angle (DbC fix)."""
    axis = np.array([0.0, 0.0, 1.0])
    with pytest.raises(TypeError):
        axis_angle_to_quaternion(axis, None)  # type: ignore[arg-type]


def test_axis_angle_to_rotation_matrix_rejects_non_numeric_angle() -> None:
    """axis_angle_to_rotation_matrix raises TypeError for non-numeric angle (DbC fix)."""
    axis = np.array([0.0, 0.0, 1.0])
    with pytest.raises(TypeError):
        axis_angle_to_rotation_matrix(axis, None)  # type: ignore[arg-type]


def test_euler_to_quaternion_rejects_non_numeric_a() -> None:
    """euler_to_quaternion raises TypeError for non-numeric angle a (DbC fix)."""
    with pytest.raises(TypeError):
        euler_to_quaternion(None, 0.0, 0.0, "xyz")  # type: ignore[arg-type]


def test_quaternion_to_euler_rejects_non_str_convention() -> None:
    """quaternion_to_euler raises TypeError for non-str convention (DbC fix)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        quaternion_to_euler(q, None)  # type: ignore[arg-type]
