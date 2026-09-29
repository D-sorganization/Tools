"""Equivalence tests for the einsum row-norm identity in electrical_model.

Locks the behavior-preserving rewrite ``np.linalg.norm(x, axis=k) ->
np.sqrt(np.einsum('...i,...i->...', x, x))`` used at every Euclidean-norm
site in electrical_model, including the calculate_system_state segment-width
row norm (axis=1, the upstream site for UpstreamDrift PR #11112's excluded
change). NaN/Inf/empty/zero semantics must be preserved up to last-ulp
float reassociation.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize(
    "shape_axis",
    [
        ((0, 3), 1),  # empty rows (num_segments == 0)
        ((1, 3), 1),
        ((30, 3), 1),  # canonical calculate_system_state shape
        ((4, 2), 1),
        ((3, 4), -1),
    ],
)
def test_row_norm_sqrt_einsum_equivalence(shape_axis) -> None:
    shape, axis = shape_axis
    rng = np.random.default_rng(1112)
    x = rng.normal(scale=1e3, size=shape)

    expected = np.linalg.norm(x, axis=axis)
    actual = np.sqrt(np.einsum("...i,...i->...", x, x))
    # Last-ulp reassociation differences are acceptable (fp only).
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_sqrt_einsum_special_values_match_norm() -> None:
    row = np.array(
        [[np.nan, 1.0, 2.0], [np.inf, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]
    )
    expected = np.linalg.norm(row, axis=1)
    actual = np.sqrt(np.einsum("...i,...i->...", row, row))
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(np.isinf(actual), np.isinf(expected))
    assert np.allclose(actual[~np.isnan(actual)], expected[~np.isnan(expected)], rtol=0.0)


def test_full_vector_norm_sqrt_einsum_equivalence() -> None:
    rng = np.random.default_rng(1113)
    for _ in range(8):
        x = rng.normal(scale=1e3, size=int(rng.integers(1, 6)))
        expected = np.linalg.norm(x)
        actual = np.sqrt(np.einsum("...i,...i->...", x, x))
        assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_empty_last_axis_shapes_match() -> None:
    x = np.zeros((4, 0))
    assert np.sqrt(np.einsum("...i,...i->...", x, x)).shape == np.linalg.norm(
        x, axis=1
    ).shape