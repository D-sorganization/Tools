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

import numpy as np
from data_processor.core.uncertainty_quantification import (
    BootstrapMethod,
    UncertaintyConfig,
    UncertaintyQuantifier,
    _leave_one_out_indices,
)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_leave_one_out_indices_match_np_delete(n: int) -> None:
    """The precomputed index matrix reproduces ``np.delete(sample, j)`` exactly.

    Vectorizing the studentized jackknife (issue #3682) must not change the set
    or order of elements in each leave-one-out subsample.
    """
    loo = _leave_one_out_indices(n)
    assert loo.shape == (n, n - 1)

    rng = np.random.default_rng(0)
    sample = rng.normal(size=n)
    for j in range(n):
        np.testing.assert_array_equal(sample[loo[j]], np.delete(sample, j))


def test_studentized_interval_matches_bruteforce_jackknife() -> None:
    """Vectorized studentized interval equals the original double-loop result.

    Reference implementation rebuilds the per-bootstrap jackknife standard error
    with the original ``np.delete``-in-a-double-loop approach on the *same*
    bootstrap samples and asserts the resulting interval is numerically
    identical (issue #3682).
    """
    config = UncertaintyConfig(
        n_bootstrap=150,
        bootstrap_method=BootstrapMethod.STUDENTIZED,
        random_seed=1234,
        confidence_level=0.9,
    )
    uq = UncertaintyQuantifier(config)
    data = np.array([2.1, 3.5, 1.9, 4.2, 3.3, 2.8, 5.1, 3.9, 2.2, 4.6])

    result = uq.bootstrap_ci(data, np.mean)

    # Brute-force reference using the exact bootstrap samples the quantifier drew.
    samples = result.bootstrap_samples
    boot_stats = result.bootstrap_statistics
    theta = float(np.mean(data))
    se = float(np.std(boot_stats, ddof=1))
    alpha = 1 - config.confidence_level

    t_stats = np.zeros(len(boot_stats))
    for i in range(len(boot_stats)):
        sample = samples[i]
        jack = np.array([np.mean(np.delete(sample, j)) for j in range(len(sample))])
        boot_se = np.std(jack) * np.sqrt(len(sample) - 1)
        t_stats[i] = (boot_stats[i] - theta) / boot_se if boot_se > 0 else 0.0

    t_lower = np.percentile(t_stats, 100 * alpha / 2)
    t_upper = np.percentile(t_stats, 100 * (1 - alpha / 2))
    ref_lower = theta - t_upper * se
    ref_upper = theta - t_lower * se

    assert result.ci_lower == pytest.approx(ref_lower, abs=1e-12, rel=0)
    assert result.ci_upper == pytest.approx(ref_upper, abs=1e-12, rel=0)


@pytest.mark.parametrize("p", [-0.1, 0.0, 1.0, 1.1])
def test_normal_ppf_rejects_out_of_range_probabilities(p: float) -> None:
    uq = UncertaintyQuantifier()

    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        uq._normal_ppf(p)


def test_normal_ppf_accepts_valid_probability() -> None:
    uq = UncertaintyQuantifier()

    assert uq._normal_ppf(0.975) == pytest.approx(1.9604, abs=0.001)
