"""Performance benchmark — Rust vs Python bilateral filter.

Tracked task: GH issue #2569. Measures the speed of the Rust-accelerated
:func:`signal_toolkit.bilateral_rust.apply_bilateral_filter_rust` against
the pure-Python reference at two signal sizes:

- 16384 samples — representative of a single-trial biomechanics window.
- 1_048_576 samples — representative of a long mocap session worth of
  marker-position data once batched across an axis.

Each run uses a deterministic ``np.random.default_rng(42).normal`` so
results are reproducible and per-run jitter only comes from the
benchmark harness itself. Skips silently if the Rust wheel is not
present.
"""

from __future__ import annotations

import numpy as np
import pytest
from signal_toolkit.core import Signal
from signal_toolkit.filters import apply_bilateral_filter

pytest.importorskip(
    "tools_core",
    reason="Rust tools_core wheel not installed on this interpreter",
)
# Imported after the skip so collection works on wheel-less machines.
from signal_toolkit.bilateral_rust import apply_bilateral_filter_rust  # noqa: E402

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


def _make_signal(n: int) -> Signal:
    rng = np.random.default_rng(42)
    return Signal(
        time=np.linspace(0.0, 1.0, n),
        values=rng.normal(0.0, 1.0, n),
        name=f"bench_{n}",
        units="au",
    )


@pytest.fixture(scope="module")
def signal_16k() -> Signal:
    return _make_signal(16_384)


@pytest.fixture(scope="module")
def signal_1m() -> Signal:
    return _make_signal(1_048_576)


# ---------------------------------------------------------------------------
# 16K samples — the "hot path" size.
# ---------------------------------------------------------------------------


def test_bilateral_python_16k(benchmark, signal_16k: Signal) -> None:
    result = benchmark(apply_bilateral_filter, signal_16k)
    assert isinstance(result, Signal)
    assert result.values.shape == signal_16k.values.shape


def test_bilateral_rust_16k(benchmark, signal_16k: Signal) -> None:
    result = benchmark(apply_bilateral_filter_rust, signal_16k)
    assert isinstance(result, Signal)
    assert result.values.shape == signal_16k.values.shape


# ---------------------------------------------------------------------------
# 1M samples — long-recording stress test. Marked slow so default `pytest`
# runs skip it; CI's benchmark stage opts in via `-m benchmark`.
# ---------------------------------------------------------------------------


def test_bilateral_rust_1m(benchmark, signal_1m: Signal) -> None:
    """Rust path on a 1M-sample signal. The Python equivalent is too slow
    to include in the default suite (≈ tens of seconds); a paired Python
    run is left as an opt-in measurement when re-baselining the speedup
    ratio reported in the PR."""
    result = benchmark(apply_bilateral_filter_rust, signal_1m)
    assert isinstance(result, Signal)
    assert result.values.shape == signal_1m.values.shape
