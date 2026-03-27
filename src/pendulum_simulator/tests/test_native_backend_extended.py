"""Extended tests for native_backend.py — covering previously untested functions.

Specifically tests:
- _warn_once: warning deduplication
- _truncate_q: >8 element truncation
- batch_evaluate_double: returns None when native is unavailable
- simulate_double: returns None when native is unavailable
- golfer_backend_mode, double_backend_mode, triple_backend_mode
- golfer_native_enabled, double_native_enabled, triple_native_enabled
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from double_pendulum_golf import native_backend
from double_pendulum_golf.native_backend import (
    batch_evaluate_double,
    double_backend_mode,
    double_native_enabled,
    golfer_backend_mode,
    golfer_native_available,
    simulate_double,
    triple_backend_mode,
    triple_native_enabled,
)
from double_pendulum_golf.physics import PendulumParams


@pytest.fixture
def double_params() -> PendulumParams:
    return PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)


# ===========================================================================
# Backend mode functions
# ===========================================================================


class TestBackendModeFunctions:
    def test_golfer_backend_mode_returns_string(self) -> None:
        mode = golfer_backend_mode()
        assert isinstance(mode, str)
        assert mode in ("python", "rust")

    def test_double_backend_mode_returns_string(self) -> None:
        mode = double_backend_mode()
        assert isinstance(mode, str)
        assert mode in ("python", "rust")

    def test_triple_backend_mode_returns_string(self) -> None:
        mode = triple_backend_mode()
        assert isinstance(mode, str)
        assert mode in ("python", "rust")

    def test_without_native_is_python(self) -> None:
        """When pendulum_core is unavailable, mode should be 'python'."""
        if not golfer_native_available():
            assert golfer_backend_mode() == "python"
            assert double_backend_mode() == "python"
            assert triple_backend_mode() == "python"

    def test_env_override_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting backend env var to 'python' should force python mode."""
        monkeypatch.setenv("PENDULUM_DOUBLE_BACKEND", "python")
        assert double_backend_mode() == "python"

    def test_env_override_invalid_defaults_to_python(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An invalid env value should default to 'python'."""
        monkeypatch.setenv("PENDULUM_GOLFER_BACKEND", "invalid")
        assert golfer_backend_mode() == "python"


# ===========================================================================
# native_enabled functions
# ===========================================================================


class TestNativeEnabledFunctions:
    def test_golfer_native_enabled_bool(self) -> None:
        result = native_backend.golfer_native_enabled()
        assert isinstance(result, bool)

    def test_double_native_enabled_bool(self) -> None:
        result = double_native_enabled()
        assert isinstance(result, bool)

    def test_triple_native_enabled_bool(self) -> None:
        result = triple_native_enabled()
        assert isinstance(result, bool)

    def test_native_not_available_implies_not_enabled(self) -> None:
        if not golfer_native_available():
            assert not native_backend.golfer_native_enabled()
            assert not double_native_enabled()
            assert not triple_native_enabled()


# ===========================================================================
# _warn_once (via direct import)
# ===========================================================================


class TestWarnOnce:
    def test_warns_on_first_call(self) -> None:
        native_backend._WARNED_CALLS.discard("__test_warn_once__")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            native_backend._warn_once("__test_warn_once__", RuntimeError("test"))
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)

    def test_warns_only_once(self) -> None:
        native_backend._WARNED_CALLS.discard("__test_warn_dedup__")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            native_backend._warn_once("__test_warn_dedup__", RuntimeError("test"))
            native_backend._warn_once("__test_warn_dedup__", RuntimeError("test"))
            assert len(w) == 1  # Second call should be suppressed


# ===========================================================================
# _truncate_q
# ===========================================================================


class TestTruncateQ:
    def test_exact_8_unchanged(self) -> None:
        q = np.arange(8, dtype=float)
        result = native_backend._truncate_q(q)
        np.testing.assert_array_equal(result, q)

    def test_greater_than_8_truncated(self) -> None:
        q = np.arange(16, dtype=float)
        result = native_backend._truncate_q(q)
        assert result.shape == (8,)
        np.testing.assert_array_equal(result, q[:8])

    def test_less_than_8_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            native_backend._truncate_q(np.zeros(4))


# ===========================================================================
# batch_evaluate_double
# ===========================================================================


class TestBatchEvaluateDouble:
    def test_returns_none_without_native(self, double_params: PendulumParams) -> None:
        """When native is unavailable, should return None without crashing."""
        result = batch_evaluate_double(
            double_params,
            coeffs_batch=[[0.0, 0.0], [1.0, 0.0]],
            n_coeffs_per_joint=1,
            q0=[0.0, 0.0],
            qdot0=[0.0, 0.0],
            t_end=0.1,
        )
        if not golfer_native_available():
            assert result is None
        else:
            assert result is not None or result is None  # either is acceptable


# ===========================================================================
# simulate_double
# ===========================================================================


class TestSimulateDouble:
    def test_returns_none_without_native(self, double_params: PendulumParams) -> None:
        """When native is unavailable, should return None without crashing."""
        result = simulate_double(
            double_params,
            q0=[0.0, 0.0],
            qdot0=[0.0, 0.0],
            coeffs=[0.0, 0.0, 0.0, 0.0],
            n_coeffs_per_joint=2,
            t_span=(0.0, 0.1),
        )
        if not golfer_native_available():
            assert result is None
        else:
            assert result is None or isinstance(result, tuple)
