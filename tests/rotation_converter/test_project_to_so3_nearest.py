"""``ProjectToSO3`` returns the *nearest* rotation, also when ``det(U @ Vh) < 0``.

The Modern Robotics port fixed a reflection by negating column 2 of ``U @ Vh``, i.e.
``U @ Vh @ diag(1, 1, -1)``. That is a rotation but not the closest one; the closest is
``U @ diag(1, 1, -1) @ Vh`` (flip the singular vector of the smallest singular value).
All three copies of the helper are checked.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

np = pytest.importorskip("numpy")

from rotation_converter import _mr_rotation_matrices, modern_robotics  # noqa: E402
from rotation_converter.modern_robotics_pkg import utils as mr_pkg_utils  # noqa: E402

IMPLS: list[Callable] = [
    _mr_rotation_matrices.ProjectToSO3,
    modern_robotics.ProjectToSO3,
    mr_pkg_utils.ProjectToSO3,
]
PROJECTORS = pytest.mark.parametrize(
    "project", IMPLS, ids=["_mr_rotation_matrices", "modern_robotics", "pkg_utils"]
)


def _rotation(rotvec: list[float]) -> np.ndarray:
    return _mr_rotation_matrices.MatrixExp3(_mr_rotation_matrices.VecToso3(rotvec))


def _is_rotation(R: np.ndarray) -> bool:
    return bool(
        np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        and abs(np.linalg.det(R) - 1.0) < 1e-9
    )


@PROJECTORS
def test_reflected_input_projects_to_the_nearest_rotation(project: Callable) -> None:
    Q = _rotation([0.3, -0.7, 0.5])
    M = np.diag([3.0, 2.0, -1.0]) @ Q  # nearest rotation is Q, at Frobenius distance 3
    R = project(M)
    assert _is_rotation(R)
    np.testing.assert_allclose(R, Q, atol=1e-9)
    assert np.linalg.norm(M - R) == pytest.approx(3.0)


@PROJECTORS
def test_no_sampled_rotation_is_closer_than_the_projection(project: Callable) -> None:
    rng = np.random.default_rng(1655)
    for _ in range(25):
        M = rng.normal(size=(3, 3))
        R = project(M)
        assert _is_rotation(R)
        best = np.linalg.norm(M - R)
        for _ in range(200):
            other = _rotation(list(rng.normal(size=3)))
            assert np.linalg.norm(M - other) >= best - 1e-9


@PROJECTORS
def test_near_rotation_input_matches_the_reference_example(project: Callable) -> None:
    mat = np.array(
        [[0.675, 0.150, 0.720], [0.370, 0.771, -0.511], [-0.630, 0.619, 0.472]]
    )
    expected = np.array(
        [
            [0.67901136, 0.14894516, 0.71885945],
            [0.37320708, 0.77319584, -0.51272279],
            [-0.63218672, 0.61642804, 0.46942137],
        ]
    )
    np.testing.assert_allclose(project(mat), expected, atol=1e-6)


@PROJECTORS
def test_input_is_not_modified(project: Callable) -> None:
    M = np.diag([3.0, 2.0, -1.0]) @ _rotation([0.1, 0.2, 0.3])
    before = M.copy()
    project(M)
    np.testing.assert_array_equal(M, before)
