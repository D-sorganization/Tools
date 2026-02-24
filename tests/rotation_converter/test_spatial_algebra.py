import math

import numpy as np

from rotation_converter.spatial_algebra import (
    FDab,
    SpatialModel,
    Xrotx,
    Xtrans,
    crf,
    crm,
)


def test_spatial_algebra_transforms() -> None:
    # Test Xtrans
    X = Xtrans([1, 2, 3])
    expected_X = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 3, -2, 1, 0, 0],
            [-3, 0, 1, 0, 1, 0],
            [2, -1, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(X, expected_X, atol=1e-10)

    # Test Xrotx
    Rx = Xrotx(math.pi / 2)
    expected_Rx = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, -1, 0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(Rx, expected_Rx, atol=1e-10)


def test_spatial_algebra_cross_products() -> None:
    v = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    M_crm = crm(v)
    M_crf = crf(v)

    # Assert relationship between crm and crf
    np.testing.assert_allclose(M_crf, -M_crm.T, atol=1e-10)


def test_fdab_simple_system() -> None:
    # 1-DOF pendulum
    NB = 1
    parent = [0]
    pitch = [0.0]  # Revolute
    Xtree = [Xtrans([0, 0, 0])]
    I_body = np.eye(6)
    I = [I_body]  # noqa: E741

    model = SpatialModel(NB=NB, parent=parent, pitch=pitch, Xtree=Xtree, I=I)

    q = [0.0]
    qd = [0.0]
    tau = [0.0]

    qdd = FDab(model, q, qd, tau)

    assert qdd.shape == (1,)
    assert not np.isnan(qdd[0])
