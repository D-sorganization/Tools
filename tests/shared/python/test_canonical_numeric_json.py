from __future__ import annotations

import pytest

from shared.python.swing_sim.canonical_numeric_json import (
    canonical_numeric_float,
    canonical_numeric_json,
)

MAX_SAFE_INTEGER = 9_007_199_254_740_991


@pytest.mark.parametrize("value", [9_007_199_254_740_992.0, 1e16, -1e20])
def test_canonical_numeric_json_rejects_unsafe_float_magnitudes(value: float) -> None:
    with pytest.raises(ValueError, match="safe range"):
        canonical_numeric_json({"value": value})
    with pytest.raises(ValueError, match="safe range"):
        canonical_numeric_float(value)


@pytest.mark.parametrize("value", [-MAX_SAFE_INTEGER, MAX_SAFE_INTEGER])
def test_canonical_numeric_json_accepts_cross_runtime_boundaries(value: int) -> None:
    expected = f'{{"float":{value},"integer":{value}}}'

    assert canonical_numeric_json({"integer": value, "float": float(value)}) == expected
    assert canonical_numeric_float(float(value)) == float(value)
