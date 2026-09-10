"""Explicit physical channel scaling and ownership for displacement transfers."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from shared.python.golf_club._shaft_transfer_ports import DisplacementPorts


def _ports() -> DisplacementPorts:
    return DisplacementPorts([[1, 2], [3, 4]], [[1, 3]], [2, 4], [0.5])


def test_normalization_is_explicit_and_does_not_assume_collocated_ports() -> None:
    inputs, outputs = _ports().normalized_arrays()
    np.testing.assert_array_equal(inputs, [[2, 8], [6, 16]])
    np.testing.assert_array_equal(outputs, [[2, 6]])


def test_inputs_and_array_results_are_owned() -> None:
    source = np.ones((2, 1))
    ports = DisplacementPorts(source, source.T, [1], [1])
    source[:] = 0
    inputs, outputs = ports.normalized_arrays()
    np.testing.assert_array_equal(inputs, [[1], [1]])
    np.testing.assert_array_equal(outputs, [[1, 1]])
    inputs[:] = 0
    outputs[:] = 0
    np.testing.assert_array_equal(ports.normalized_arrays()[0], [[1], [1]])
    with pytest.raises(FrozenInstanceError):
        ports.input_scales = (0,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("force_map", []),
        ("force_map", [1, 2]),
        ("force_map", [[1, 2, 3]]),
        ("observation_map", [[1]]),
        ("observation_map", np.zeros((0, 2))),
        ("input_scales", [1]),
        ("input_scales", [1, 0]),
        ("output_scales", [-1]),
        ("force_map", [[1, True], [3, 4]]),
        ("observation_map", [["1", "3"]]),
        ("input_scales", [1j, 1]),
        ("output_scales", [float("nan")]),
    ],
)
def test_malformed_or_nonphysical_channel_scales_are_refused(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_ports(), **{field: value})


@pytest.mark.parametrize("force,scale", [(1e308, 1e308), (1e-200, 1e-200)])
def test_unrepresentable_normalization_is_refused(force: float, scale: float) -> None:
    with pytest.raises(ValueError, match="representable"):
        DisplacementPorts([[force]], [[1]], [scale], [1])
