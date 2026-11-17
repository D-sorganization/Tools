"""Invalid shared seeds must not partially change random generator state."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import cast

import numpy as np
import pytest

from logging_pkg import set_seeds

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def preserve_rng_state() -> Iterator[None]:
    """Keep validation checks from changing subsequent tests' random streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


@pytest.mark.parametrize("seed", [2**32, 2**40, 1.5])
def test_invalid_seed_preserves_both_random_streams(seed: object) -> None:
    """Default validation rejects bad seeds before either backend is mutated."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with pytest.raises((TypeError, ValueError)):
        set_seeds(cast(int, seed))
    assert random.getstate() == python_state
    actual = np.random.get_state()
    assert actual[0] == numpy_state[0]
    np.testing.assert_array_equal(actual[1], numpy_state[1])
    assert actual[2:] == numpy_state[2:]


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_common_seed_boundaries_reproduce_both_streams(seed: int) -> None:
    """The full shared unsigned-32-bit domain remains usable."""
    set_seeds(seed)
    python_sample = random.random()
    numpy_sample = np.random.random(3)
    set_seeds(seed)
    assert random.random() == python_sample
    np.testing.assert_array_equal(np.random.random(3), numpy_sample)
