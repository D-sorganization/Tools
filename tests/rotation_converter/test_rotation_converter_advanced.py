from __future__ import annotations

import math

import numpy as np

from rotation_converter.advanced_kinematics import DualQuaternion, dh_to_matrix, slerp
from rotation_converter.core import normalize_quaternion


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
    q2 = normalize_quaternion(np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]))

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
