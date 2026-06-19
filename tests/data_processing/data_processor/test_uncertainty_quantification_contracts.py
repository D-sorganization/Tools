import sys
import types

import pytest

numba_stub = types.ModuleType("numba")


def _jit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    def decorator(func):
        return func

    return decorator


numba_stub.jit = _jit
sys.modules.setdefault("numba", numba_stub)

from data_processor.core.uncertainty_quantification import UncertaintyQuantifier


@pytest.mark.parametrize("p", [-0.1, 0.0, 1.0, 1.1])
def test_normal_ppf_rejects_out_of_range_probabilities(p: float) -> None:
    uq = UncertaintyQuantifier()

    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        uq._normal_ppf(p)


def test_normal_ppf_accepts_valid_probability() -> None:
    uq = UncertaintyQuantifier()

    assert uq._normal_ppf(0.975) == pytest.approx(1.9604, abs=0.001)
