"""Tests for 6x6 spatial inertia utilities.

These exercise the algebraic round-trips (URDF <-> spatial inertia),
the structural form of the spatial transform, and the validation
contract on matrix shapes.
"""

from __future__ import annotations

import numpy as np
import pytest
from model_generation.inertia import spatial as S


class TestMcI:
    def test_shape_and_mass_block(self) -> None:
        M = S.mcI(2.0, [0.0, 0.0, 0.0], np.diag([0.1, 0.2, 0.3]))
        assert M.shape == (6, 6)
        # Lower-right 3x3 block is m * I3.
        assert np.allclose(M[3:, 3:], 2.0 * np.eye(3))

    def test_zero_com_gives_block_diagonal(self) -> None:
        I_com = np.diag([0.1, 0.2, 0.3])
        M = S.mcI(2.0, np.zeros(3), I_com)
        # With COM at origin, the coupling blocks vanish and the
        # rotational block equals I_com.
        assert np.allclose(M[:3, 3:], 0.0)
        assert np.allclose(M[:3, :3], I_com)

    def test_symmetric(self) -> None:
        M = S.mcI(1.5, [0.1, -0.2, 0.05], np.diag([0.3, 0.4, 0.5]))
        assert np.allclose(M, M.T)

    def test_asymmetric_inertia_is_symmetrized(self) -> None:
        asym = np.array([[0.1, 0.05, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
        M = S.mcI(1.0, np.zeros(3), asym)
        # The off-diagonal becomes the average of the two entries.
        assert M[0, 1] == pytest.approx(0.025)
        assert M[1, 0] == pytest.approx(0.025)

    def test_bad_com_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            S.mcI(1.0, [1.0, 2.0], np.eye(3))

    def test_bad_inertia_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            S.mcI(1.0, [1.0, 2.0, 3.0], np.eye(2))


class TestUrdfRoundTrip:
    def test_mcI_recovers_urdf_parameters(self) -> None:
        mass = 2.0
        com = np.array([0.05, -0.01, 0.1])
        M = S.mcI(mass, com, np.diag([0.1, 0.2, 0.3]))
        out = S.spatial_inertia_to_urdf(M)

        assert out["mass"] == pytest.approx(mass)
        assert out["com"] == pytest.approx([0.05, -0.01, 0.1])
        assert out["ixx"] == pytest.approx(0.1)
        assert out["iyy"] == pytest.approx(0.2)
        assert out["izz"] == pytest.approx(0.3)

    def test_urdf_to_spatial_matches_mcI(self) -> None:
        com = [0.05, 0.0, 0.1]
        direct = S.mcI(2.0, np.asarray(com), np.diag([0.1, 0.2, 0.3]))
        via_urdf = S.urdf_to_spatial_inertia(2.0, com, 0.1, 0.2, 0.3)
        assert np.allclose(direct, via_urdf)

    def test_to_urdf_rejects_nonpositive_mass(self) -> None:
        bad = np.zeros((6, 6))
        with pytest.raises(ValueError):
            S.spatial_inertia_to_urdf(bad)

    def test_to_urdf_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            S.spatial_inertia_to_urdf(np.eye(5))

    def test_urdf_to_spatial_none_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            S.urdf_to_spatial_inertia(None, [0, 0, 0], 0.1, 0.1, 0.1)  # type: ignore[arg-type]


class TestSpatialTransform:
    def test_identity_transform_structure(self) -> None:
        X = S.spatial_transform(np.eye(3), np.zeros(3))
        assert np.allclose(X, np.eye(6))

    def test_pure_rotation_block_structure(self) -> None:
        # 90-degree rotation about z; zero translation -> block-diagonal R.
        theta = np.pi / 2
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        X = S.spatial_transform(R, np.zeros(3))
        assert np.allclose(X[:3, :3], R)
        assert np.allclose(X[3:, 3:], R)
        assert np.allclose(X[3:, :3], 0.0)

    def test_translation_fills_lower_left_block(self) -> None:
        X = S.spatial_transform(np.eye(3), [1.0, 0.0, 0.0])
        # Lower-left block is t_skew @ R; should be non-zero with translation.
        assert not np.allclose(X[3:, :3], 0.0)

    def test_bad_rotation_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            S.spatial_transform(np.eye(2), np.zeros(3))

    def test_bad_translation_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            S.spatial_transform(np.eye(3), [1.0, 2.0])


class TestTransformSpatialInertia:
    def test_identity_preserves_inertia(self) -> None:
        M = S.mcI(2.0, [0.0, 0.0, 0.0], np.diag([0.1, 0.2, 0.3]))
        X = S.spatial_transform(np.eye(3), np.zeros(3))
        assert np.allclose(S.transform_spatial_inertia(M, X), M)

    def test_congruence_preserves_symmetry(self) -> None:
        M = S.mcI(1.5, [0.1, 0.0, 0.2], np.diag([0.2, 0.3, 0.4]))
        R = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        X = S.spatial_transform(R, [0.0, 0.0, 0.0])
        result = S.transform_spatial_inertia(M, X)
        assert np.allclose(result, result.T)

    def test_bad_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            S.transform_spatial_inertia(np.eye(5), np.eye(6))


class TestComposite:
    def test_empty_list_is_zero(self) -> None:
        assert np.allclose(S.composite_rigid_body_inertia([]), np.zeros((6, 6)))

    def test_single_identity_transform_equals_input(self) -> None:
        M = S.mcI(2.0, [0.0, 0.0, 0.0], np.diag([0.1, 0.2, 0.3]))
        X = S.spatial_transform(np.eye(3), np.zeros(3))
        assert np.allclose(S.composite_rigid_body_inertia([(M, X)]), M)

    def test_two_bodies_sum(self) -> None:
        X = S.spatial_transform(np.eye(3), np.zeros(3))
        M1 = S.mcI(1.0, np.zeros(3), np.diag([0.1, 0.1, 0.1]))
        M2 = S.mcI(2.0, np.zeros(3), np.diag([0.2, 0.2, 0.2]))
        combined = S.composite_rigid_body_inertia([(M1, X), (M2, X)])
        assert np.allclose(combined, M1 + M2)
