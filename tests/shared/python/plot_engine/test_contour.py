import numpy as np
import pytest
from plot_engine.contour import correlation_matrix, scatter_to_grid


def test_scatter_to_grid_interpolates_square_points_to_requested_resolution() -> None:
    x_grid, y_grid, z_grid = scatter_to_grid(
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        resolution=2,
        method="linear",
    )

    assert np.array_equal(x_grid, np.array([0.0, 1.0]))
    assert np.array_equal(y_grid, np.array([0.0, 1.0]))
    assert z_grid.shape == (2, 2)
    assert np.allclose(z_grid, np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_scatter_to_grid_filters_nan_entries_before_interpolation() -> None:
    x_grid, y_grid, z_grid = scatter_to_grid(
        np.array([0.0, 1.0, 0.0, 1.0, np.nan]),
        np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 2.0, 3.0, 99.0]),
        resolution=2,
        method="nearest",
    )

    assert np.array_equal(x_grid, np.array([0.0, 1.0]))
    assert np.array_equal(y_grid, np.array([0.0, 1.0]))
    assert np.allclose(z_grid, np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_scatter_to_grid_rejects_fewer_than_three_valid_points() -> None:
    with pytest.raises(
        ValueError,
        match="At least 3 valid points required for grid interpolation",
    ):
        scatter_to_grid(
            np.array([0.0, 1.0, np.nan]),
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        )


def test_correlation_matrix_uses_default_labels_and_matches_numpy() -> None:
    data = np.array(
        [
            [1.0, 10.0, 2.0],
            [2.0, 20.0, 1.0],
            [3.0, 30.0, 0.0],
            [4.0, 40.0, -1.0],
        ]
    )

    corr, labels = correlation_matrix(data)

    assert labels == ["Var 0", "Var 1", "Var 2"]
    assert np.allclose(corr, np.corrcoef(data, rowvar=False))
    assert np.allclose(np.diag(corr), np.ones(3))


def test_correlation_matrix_returns_custom_labels_unmodified() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])
    labels = ["feed", "output"]

    corr, returned_labels = correlation_matrix(data, labels=labels)

    assert returned_labels is labels
    assert corr.shape == (2, 2)
    assert np.allclose(corr, np.corrcoef(data, rowvar=False))


@pytest.mark.parametrize(
    "data",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([[[1.0], [2.0]]]),
    ],
)
def test_correlation_matrix_requires_2d_data(data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Data must be 2D"):
        correlation_matrix(data)
