"""Independent analytic and metamorphic contracts for triangle-area mass."""

import numpy as np
import pytest
from numpy.typing import NDArray

from shared.python.humanoid_character_builder.mesh.triangle_lamina import (
    compute_triangle_lamina,
)


def triangle() -> NDArray[np.float64]:
    return np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 6.0, 0.0]])


def test_triangle_analytic_mass_com_and_off_diagonal() -> None:
    result = compute_triangle_lamina(triangle(), np.array([[0, 1, 2]]), 2.0)
    assert result.inertia.center_of_mass == pytest.approx((1.0, 2.0, 0.0))
    np.testing.assert_allclose(
        result.inertia.as_matrix(), [[4.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 5.0]]
    )
    assert result.total_area_m2 == pytest.approx(9.0)
    assert result.inertia.mass == 2.0
    assert result.inertia.volume == 0.0
    assert result.inertia.mode.value == "triangle_multiset_lamina_mass"
    assert result.boundary_edges == 3


def test_rectangular_plate_and_subdivision() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 6.0, 0.0], [0.0, 6.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    result = compute_triangle_lamina(vertices, faces, 3.0)
    np.testing.assert_allclose(
        result.inertia.as_matrix(), np.diag([9.0, 4.0, 13.0]), atol=1e-14
    )
    center = vertices.mean(axis=0)
    subdivided = compute_triangle_lamina(
        np.vstack([vertices, center]),
        np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
        3.0,
    )
    np.testing.assert_allclose(
        subdivided.inertia.as_matrix(), result.inertia.as_matrix(), atol=1e-14
    )
    assert subdivided.inertia.center_of_mass == pytest.approx(
        result.inertia.center_of_mass
    )


def test_box_shell_analytic() -> None:
    # Cube side 2: two faces contribute E[x²]=1, four contribute E[x²]=1/3.
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    result = compute_triangle_lamina(vertices, faces, 9.0)
    assert result.total_area_m2 == pytest.approx(24.0)
    np.testing.assert_allclose(result.inertia.as_matrix(), np.eye(3) * 10.0, atol=1e-14)
    assert result.boundary_edges == result.nonmanifold_edges == 0


def test_rotation_translation_scaling_and_no_input_mutation() -> None:
    vertices, faces = triangle(), np.array([[0, 1, 2]])
    original_vertices, original_faces = vertices.copy(), faces.copy()
    reference = compute_triangle_lamina(vertices, faces, 2.0)
    angle = 0.47
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = compute_triangle_lamina(
        vertices @ rotation.T + [8.0, -3.0, 7.0], faces[:, ::-1], 2.0
    )
    np.testing.assert_allclose(
        transformed.inertia.as_matrix(),
        rotation @ reference.inertia.as_matrix() @ rotation.T,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        transformed.inertia.center_of_mass,
        rotation @ np.array(reference.inertia.center_of_mass) + [8.0, -3.0, 7.0],
    )
    scaled = compute_triangle_lamina(vertices * 4, faces, 6.0)
    np.testing.assert_allclose(
        scaled.inertia.as_matrix(), reference.inertia.as_matrix() * 48
    )
    translated = compute_triangle_lamina(vertices + 2.0**40, faces, 2.0)
    np.testing.assert_allclose(
        translated.inertia.as_matrix(), reference.inertia.as_matrix(), atol=1e-14
    )
    np.testing.assert_array_equal(vertices, original_vertices)
    np.testing.assert_array_equal(faces, original_faces)


def test_multiset_duplicates_and_exact_zero_area_are_reported() -> None:
    vertices = np.vstack([triangle(), triangle()])
    result = compute_triangle_lamina(
        vertices, np.array([[0, 1, 2], [5, 4, 3], [0, 0, 1]]), 2.0
    )
    assert result.total_area_m2 == 18.0
    assert result.duplicate_positive_area_triangles == 1
    assert result.zero_area_triangles == 1
    np.testing.assert_allclose(
        result.inertia.as_matrix(), [[4.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 5.0]]
    )


def test_distant_zero_mass_face_cannot_change_reference_origin() -> None:
    vertices = np.vstack([triangle(), np.full((3, 3), 1e16)])
    expected = compute_triangle_lamina(vertices, np.array([[0, 1, 2]]), 2.0)
    actual = compute_triangle_lamina(vertices, np.array([[3, 4, 5], [0, 1, 2]]), 2.0)
    np.testing.assert_array_equal(
        actual.inertia.as_matrix(), expected.inertia.as_matrix()
    )
    assert actual.inertia.center_of_mass == expected.inertia.center_of_mass
    assert actual.zero_area_triangles == 1


def test_small_representable_area_is_not_mislabeled_zero() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 1e-100, 0.0], [0.0, 0.0, 1e-100]])
    result = compute_triangle_lamina(vertices, np.array([[0, 1, 2]]), 2.0)
    assert result.zero_area_triangles == 0
    assert result.total_area_m2 == pytest.approx(5e-201, rel=1e-14, abs=0.0)
    np.testing.assert_allclose(
        result.inertia.as_matrix(),
        np.array([[2 / 9, 0, 0], [0, 1 / 9, 1 / 18], [0, 1 / 18, 1 / 9]]) * 1e-200,
        rtol=1e-14,
        atol=0.0,
    )


def test_negligible_remote_face_cannot_change_origin_when_reordered() -> None:
    tiny = np.array([[1e100, 0.0, 0.0], [1e100, 1e-100, 0.0], [1e100, 0.0, 1e-100]])
    vertices = np.vstack([triangle(), tiny])
    normal_first = compute_triangle_lamina(
        vertices, np.array([[0, 1, 2], [3, 4, 5]]), 2.0
    )
    tiny_first = compute_triangle_lamina(
        vertices, np.array([[3, 4, 5], [0, 1, 2]]), 2.0
    )
    np.testing.assert_allclose(
        tiny_first.inertia.as_matrix(), normal_first.inertia.as_matrix(), rtol=1e-14
    )
    np.testing.assert_allclose(
        tiny_first.inertia.center_of_mass,
        normal_first.inertia.center_of_mass,
        rtol=1e-14,
    )


def test_full_three_dimensional_tensor_covariance() -> None:
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    original = compute_triangle_lamina(triangle(), faces, 2.0)
    transformed = compute_triangle_lamina(triangle() @ rotation.T, faces, 2.0)
    np.testing.assert_allclose(
        transformed.inertia.as_matrix(),
        rotation @ original.inertia.as_matrix() @ rotation.T,
        atol=0.0,
    )
    assert transformed.inertia.iyz == 1.0


def test_triangle_multiplicity_changes_mass_distribution_explicitly() -> None:
    vertices = np.vstack([triangle(), triangle() + [3.0, 0.0, 0.0]])
    result = compute_triangle_lamina(
        vertices, np.array([[0, 1, 2], [0, 1, 2], [3, 4, 5]]), 3.0
    )
    np.testing.assert_allclose(result.inertia.center_of_mass, [2.0, 2.0, 0.0])
    np.testing.assert_allclose(
        result.inertia.as_matrix(),
        [[6.0, 1.5, 0.0], [1.5, 7.5, 0.0], [0.0, 0.0, 13.5]],
        atol=1e-14,
    )
    assert result.duplicate_positive_area_triangles == 1


def test_unrepresentable_area_fails_instead_of_becoming_zero_mass() -> None:
    tiny = np.array([[0.0, 0.0, 0.0], [0.0, 1e-200, 0.0], [0.0, 0.0, 1e-200]])
    vertices = np.vstack([triangle(), tiny])
    with pytest.raises(ValueError, match="area.*represent"):
        compute_triangle_lamina(vertices, np.array([[0, 1, 2], [3, 4, 5]]), 2.0)


def test_complex_vertices_cannot_silently_discard_imaginary_coordinates() -> None:
    vertices = triangle().astype(complex)
    vertices[1, 0] += 5j
    with pytest.raises(ValueError, match="vertices.*real"):
        compute_triangle_lamina(vertices, np.array([[0, 1, 2]]), 2.0)


@pytest.mark.parametrize("mass", [0.0, -1.0, np.inf, np.nan, True])
def test_invalid_mass(mass: float) -> None:
    with pytest.raises((TypeError, ValueError), match="mass"):
        compute_triangle_lamina(triangle(), np.array([[0, 1, 2]]), mass)


@pytest.mark.parametrize(
    "faces",
    [
        np.array([[0.0, 1.0, 2.0]]),
        np.array([[True, False, True]]),
        np.array([[0, 1, 3]]),
        np.array([[-1, 1, 2]]),
        np.array([[0, 1]]),
        np.empty((0, 3), dtype=int),
    ],
)
def test_invalid_faces(faces: NDArray[np.integer]) -> None:
    with pytest.raises((TypeError, ValueError), match="faces|area"):
        compute_triangle_lamina(triangle(), faces, 2.0)


@pytest.mark.parametrize(
    "vertices",
    [
        np.zeros((3, 3)),
        np.ones((3, 2)),
        np.array([[np.nan, 0, 0], [1, 0, 0], [0, 1, 0]]),
        np.array([[np.inf, 0, 0], [1, 0, 0], [0, 1, 0]]),
    ],
)
def test_invalid_vertices(vertices: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="vertices|area"):
        compute_triangle_lamina(vertices, np.array([[0, 1, 2]]), 2.0)
