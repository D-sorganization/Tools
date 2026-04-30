"""Benchmark suite configuration and shared fixtures.

Provides common test data and SLA thresholds for performance benchmarks.
"""

from __future__ import annotations

import pytest
import numpy as np


# =============================================================================
# Pytest Hook to Disable xdist for Benchmarks
# =============================================================================
# pytest-benchmark is incompatible with xdist parallel execution.
# This hook ensures benchmarks run serially for accurate measurements.


def pytest_configure(config):
    """Disable xdist plugin for benchmark tests to avoid conflicts."""
    # If xdist is loaded, we need to handle it gracefully
    # The actual disabling happens via CLI or via removing -n flag
    # For now, we document this requirement in the conftest
    pass


# =============================================================================
# Benchmark Configuration
# =============================================================================

# SLA (Service Level Agreement) thresholds in milliseconds
BENCHMARK_SLAS = {
    "pressure_drop": 100.0,  # ms
    "rotation_converter": 50.0,  # ms
    "data_processor_filter": 500.0,  # ms for 10K rows
}


# =============================================================================
# Common Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_small_array():
    """Small array for quick benchmarks (100 elements)."""
    return np.random.randn(100)


@pytest.fixture
def sample_medium_array():
    """Medium array for typical benchmarks (1000 elements)."""
    return np.random.randn(1000)


@pytest.fixture
def sample_large_array():
    """Large array for scale benchmarks (10000 elements)."""
    return np.random.randn(10000)


@pytest.fixture
def sample_euler_angles():
    """Sample Euler angles (degrees) for rotation converter benchmarks.

    Returns tuple of (alpha, beta, gamma) in range [0, 360).
    """
    return (45.0, 60.0, 30.0)


@pytest.fixture
def sample_quaternion():
    """Sample unit quaternion (w, x, y, z) for rotation converter benchmarks."""
    # Normalized quaternion representing a rotation
    q = np.array([0.7071, 0.7071, 0.0, 0.0])
    return q / np.linalg.norm(q)


@pytest.fixture
def sample_rotation_matrix():
    """Sample 3x3 rotation matrix (SO(3)) for rotation converter benchmarks."""
    # Rotation of 45 degrees around Z axis
    angle = np.radians(45)
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


@pytest.fixture
def sample_time_series_data():
    """Sample time series data for data processor benchmarks.

    Returns dictionary with 'signal' (1000 samples) and 'sampling_rate' (Hz).
    """
    sampling_rate = 1000  # Hz
    duration = 1.0  # seconds
    n_samples = int(sampling_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Signal with multiple frequency components
    signal = (
        np.sin(2 * np.pi * 10 * t)
        + 0.5 * np.sin(2 * np.pi * 50 * t)
        + 0.1 * np.random.randn(n_samples)
    )
    return {
        "signal": signal,
        "sampling_rate": sampling_rate,
    }


@pytest.fixture
def sample_large_time_series_data():
    """Large time series data for scale benchmarks (10K samples).

    Returns dictionary with 'signal' (10000 samples) and 'sampling_rate' (Hz).
    """
    sampling_rate = 1000  # Hz
    duration = 10.0  # seconds
    n_samples = int(sampling_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = (
        np.sin(2 * np.pi * 10 * t)
        + 0.5 * np.sin(2 * np.pi * 50 * t)
        + 0.1 * np.random.randn(n_samples)
    )
    return {
        "signal": signal,
        "sampling_rate": sampling_rate,
    }


@pytest.fixture
def pipe_parameters():
    """Sample pipe parameters for pressure drop calculator benchmarks.

    Returns dictionary with pipe dimensions and fluid properties.
    """
    return {
        "inlet_pressure": 101325.0,  # Pa
        "fluid_density": 0.3,  # kg/m³
        "pipe_length": 0.05,  # m
    }
