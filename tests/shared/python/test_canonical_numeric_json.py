from __future__ import annotations

import pytest

from shared.python.swing_sim.canonical_numeric_json import (
    canonical_numeric_float,
    canonical_numeric_json,
    canonical_numeric_json_extended_floats,
)

MAX_SAFE_INTEGER = 9_007_199_254_740_991

#: Floats that a JavaScript ``Number`` cannot hold exactly, so they cannot cross
#: the Python/browser boundary without silently changing value.
BEYOND_SAFE_RANGE = [9_007_199_254_740_992.0, 1e16, -1e20, 1e21]


@pytest.mark.parametrize("value", BEYOND_SAFE_RANGE)
def test_default_encoder_fails_closed_beyond_the_cross_runtime_safe_range(
    value: float,
) -> None:
    """The default contract refuses magnitudes a JS ``Number`` cannot represent.

    ``canonical_numeric_json`` encodes browser-facing payloads (the
    regional-ground authority job status wire), where a value above 2**53-1
    would arrive in the browser as a different number. Failing closed is the
    contract; callers that legitimately hold larger floats opt in explicitly
    through ``canonical_numeric_json_extended_floats``.
    """
    with pytest.raises(ValueError, match="cross-runtime safe range"):
        canonical_numeric_json(value)
    with pytest.raises(ValueError, match="cross-runtime safe range"):
        canonical_numeric_float(value)


@pytest.mark.parametrize("value", BEYOND_SAFE_RANGE)
def test_extended_encoder_retains_the_established_large_float_domain(
    value: float,
) -> None:
    """The opt-in encoder still serializes the pre-guard float domain exactly.

    This is the path the capability-observation facade re-exports, whose
    payloads carry declared magnitudes such as 1e20 and 1e21 and must not
    degrade to exponent notation.
    """
    encoded = canonical_numeric_json_extended_floats(value)

    assert encoded == str(int(value))
    assert "e" not in encoded


def test_extended_encoder_still_refuses_unsafe_integers() -> None:
    """Widening applies to floats only; ints beyond the safe range still fail."""
    with pytest.raises(ValueError, match="integer exceeds cross-runtime safe range"):
        canonical_numeric_json_extended_floats(MAX_SAFE_INTEGER + 1)


@pytest.mark.parametrize("value", [-MAX_SAFE_INTEGER, MAX_SAFE_INTEGER])
def test_canonical_numeric_json_accepts_cross_runtime_boundaries(value: int) -> None:
    expected = f'{{"float":{value},"integer":{value}}}'

    assert canonical_numeric_json({"integer": value, "float": float(value)}) == expected
    assert canonical_numeric_float(float(value)) == float(value)
