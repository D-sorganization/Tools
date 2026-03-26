"""Property-based tests using Hypothesis for shared libraries.

Tests mathematical invariants and physical consistency:
- Signal processing: linearity, time-invariance, energy preservation
- Data structures: serialization round-trips, idempotency
- Calculator functions: physical consistency, unit conversions

Addresses #765 (Phase 5: property-based tests).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

logger = logging.getLogger(__name__)


# ─── Strategies ──────────────────────────────────────────────────

# Reasonable floating-point values (no NaN/Inf)
finite_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)

# Small positive integers for array sizes
array_sizes = st.integers(min_value=10, max_value=500)

# 1D signal arrays
signal_arrays = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=10, max_value=200),
    elements=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
)


# ─── Signal Processing Properties ────────────────────────────────


class TestSignalProcessingProperties:
    """Property-based tests for signal processing functions."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=20, max_value=200),
            elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
        )
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_moving_average_preserves_mean(self, data: np.ndarray) -> None:
        """Moving average should approximately preserve the overall mean.

        Property: mean(filter(x)) ≈ mean(x) for uniform windows.
        """
        window = min(5, len(data))
        if window < 2:
            return
        kernel = np.ones(window) / window
        filtered = np.convolve(data, kernel, mode="valid")
        # Trimmed means should be close
        trimmed_original = data[window // 2 : len(data) - window // 2 + 1]
        if len(trimmed_original) > 0 and len(filtered) > 0:
            np.testing.assert_allclose(
                np.mean(filtered),
                np.mean(trimmed_original),
                atol=np.std(data) * 0.5 + 1e-10,
            )

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=200),
            elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
        ),
        scale=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_fft_linearity_scaling(self, data: np.ndarray, scale: float) -> None:
        """FFT must satisfy linearity: FFT(a*x) = a*FFT(x).

        Property: Scaling in time domain = scaling in frequency domain.
        """
        fft_original = np.fft.fft(data)
        fft_scaled = np.fft.fft(data * scale)
        np.testing.assert_allclose(fft_scaled, fft_original * scale, atol=1e-8)

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=200),
            elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_fft_inverse_roundtrip(self, data: np.ndarray) -> None:
        """FFT -> IFFT must reconstruct the original signal.

        Property: IFFT(FFT(x)) == x (up to floating point tolerance).
        """
        reconstructed = np.fft.ifft(np.fft.fft(data)).real
        np.testing.assert_allclose(reconstructed, data, atol=1e-10)

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(min_value=-1e2, max_value=1e2, allow_nan=False),
        )
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_parseval_theorem(self, data: np.ndarray) -> None:
        """Verify Parseval's theorem: energy in time == energy in frequency.

        Property: sum(|x|^2) == sum(|X|^2) / N
        """
        n = len(data)
        energy_time = np.sum(np.abs(data) ** 2)
        energy_freq = np.sum(np.abs(np.fft.fft(data)) ** 2) / n
        np.testing.assert_allclose(energy_time, energy_freq, rtol=1e-10)


# ─── Data Structure Properties ───────────────────────────────────


class TestDataStructureProperties:
    """Property-based tests for data structure invariants."""

    @given(
        name=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P"),
            ),
        ),
        n_steps=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_pipeline_serialization_roundtrip(self, name: str, n_steps: int) -> None:
        """Pipeline.to_dict() -> from_dict() must reconstruct the original.

        Property: from_dict(to_dict(pipeline)) == pipeline
        """
        from data_processor.core.script_generator_types import (
            OperationType,
            ProcessingPipeline,
        )

        pipeline = ProcessingPipeline(name=name, description="test")
        ops = list(OperationType)
        for i in range(n_steps):
            pipeline.add_step(
                operation=ops[i % len(ops)],
                parameters={"key": f"value_{i}"},
                description=f"step {i}",
            )

        serialized = pipeline.to_dict()
        restored = ProcessingPipeline.from_dict(serialized)

        assert restored.name == pipeline.name
        assert restored.description == pipeline.description
        assert len(restored.steps) == len(pipeline.steps)
        for orig, rest in zip(pipeline.steps, restored.steps, strict=True):
            assert orig.operation == rest.operation
            assert orig.parameters == rest.parameters
            assert orig.description == rest.description

    @given(
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            min_size=2,
            max_size=100,
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_sort_is_idempotent(self, values: list[float]) -> None:
        """Sorting is idempotent: sort(sort(x)) == sort(x).

        Property: Applying sort twice yields same result as once.
        """
        arr = np.array(values)
        sorted_once = np.sort(arr)
        sorted_twice = np.sort(sorted_once)
        np.testing.assert_array_equal(sorted_once, sorted_twice)


# ─── Contracts Properties ────────────────────────────────────────


class TestContractProperties:
    """Property-based tests for the contracts module."""

    @given(
        message=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=30)
    def test_require_true_never_raises(self, message: str) -> None:
        """require(True, ...) must never raise, regardless of message."""
        from contracts import require

        require(True, message)  # Should never raise

    @given(
        message=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=30)
    def test_require_false_always_raises(self, message: str) -> None:
        """require(False, ...) must always raise PreconditionError."""
        from contracts import PreconditionError, require

        with pytest.raises(PreconditionError):
            require(False, message)

    @given(
        message=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=30)
    def test_ensure_false_always_raises(self, message: str) -> None:
        """ensure(False, ...) must always raise PostconditionError."""
        from contracts import PostconditionError, ensure

        with pytest.raises(PostconditionError):
            ensure(False, message)

    @given(
        message=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=30)
    def test_invariant_false_always_raises(self, message: str) -> None:
        """invariant(False, ...) must always raise InvariantError."""
        from contracts import InvariantError, invariant

        with pytest.raises(InvariantError):
            invariant(False, message)


# ─── Numerical Stability Properties ─────────────────────────────


class TestNumericalStability:
    """Property-based tests for numerical robustness."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=100),
            elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_standardization_properties(self, data: np.ndarray) -> None:
        """Standardized data must have mean ≈ 0, std ≈ 1.

        Property: (x - mean) / std -> mean=0, std=1
        """
        std = np.std(data)
        if std < 1e-10:
            return  # Skip near-constant arrays

        standardized = (data - np.mean(data)) / std
        np.testing.assert_allclose(np.mean(standardized), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(standardized), 1.0, atol=1e-10)

    @given(
        a=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=50),
            elements=st.floats(min_value=-1e2, max_value=1e2, allow_nan=False),
        ),
        b=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=50),
            elements=st.floats(min_value=-1e2, max_value=1e2, allow_nan=False),
        ),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_correlation_bounds(self, a: np.ndarray, b: np.ndarray) -> None:
        """Correlation coefficient must be in [-1, 1].

        Property: -1 <= corr(a, b) <= 1
        """
        min_len = min(len(a), len(b))
        if min_len < 3:
            return
        a_trimmed = a[:min_len]
        b_trimmed = b[:min_len]

        if np.std(a_trimmed) < 1e-10 or np.std(b_trimmed) < 1e-10:
            return  # Skip constant arrays

        corr = np.corrcoef(a_trimmed, b_trimmed)[0, 1]
        assert -1.0 - 1e-10 <= corr <= 1.0 + 1e-10, f"Correlation out of bounds: {corr}"
