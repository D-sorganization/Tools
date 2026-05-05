"""Performance benchmarks for rotation converter.

Measures the performance of rotation representation conversions.
Note: rotation_converter is deprecated in favor of tools_core.math_primitives.
SLA target: < 50ms per conversion.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# Suppress deprecation warning for this test suite
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from rotation_converter.core import (
            euler_to_quaternion,
            quaternion_to_rotation_matrix,
            rotation_matrix_to_quaternion,
            quaternion_to_axis_angle,
            axis_angle_to_quaternion,
            quaternion_multiply,
            normalize_quaternion,
        )
    except (ImportError, NameError):
        pytest.skip(
            "rotation_converter not available",
            allow_module_level=True,
        )


pytestmark = pytest.mark.benchmark


@pytest.mark.performance
class TestRotationConverterBasic:
    """Performance benchmarks for basic rotation conversions."""

    def test_euler_to_quaternion_conversion(
        self, benchmark, sample_euler_angles
    ):
        """Benchmark Euler angles to quaternion conversion.

        SLA: < 50ms
        Tests: Single conversion from Euler XYZ convention
        """
        alpha, beta, gamma = sample_euler_angles
        result = benchmark(
            euler_to_quaternion, alpha, beta, gamma, convention="xyz"
        )
        assert result is not None
        assert len(result) == 4

    def test_quaternion_to_rotation_matrix_conversion(
        self, benchmark, sample_quaternion
    ):
        """Benchmark quaternion to rotation matrix conversion.

        SLA: < 50ms
        Tests: Single conversion to 3x3 SO(3) matrix
        """
        result = benchmark(
            quaternion_to_rotation_matrix, sample_quaternion
        )
        assert result.shape == (3, 3)
        # Verify orthogonality
        assert np.allclose(
            result @ result.T, np.eye(3), atol=1e-6
        )

    def test_rotation_matrix_to_quaternion_conversion(
        self, benchmark, sample_rotation_matrix
    ):
        """Benchmark rotation matrix to quaternion conversion.

        SLA: < 50ms
        Tests: Single conversion from 3x3 matrix
        """
        result = benchmark(
            rotation_matrix_to_quaternion, sample_rotation_matrix
        )
        assert result is not None
        assert len(result) == 4

    def test_axis_angle_to_quaternion_conversion(
        self, benchmark
    ):
        """Benchmark axis-angle to quaternion conversion.

        SLA: < 50ms
        Tests: Single conversion from unit axis and angle
        """
        axis = np.array([0.0, 0.0, 1.0])  # Z axis
        angle = np.radians(45)

        result = benchmark(
            axis_angle_to_quaternion, axis, angle
        )
        assert result is not None
        assert len(result) == 4


@pytest.mark.performance
class TestRotationConverterChains:
    """Performance benchmarks for chained conversions."""

    def test_euler_to_matrix_chain(
        self, benchmark, sample_euler_angles
    ):
        """Benchmark Euler to matrix conversion chain.

        SLA: < 50ms
        Tests: Euler -> Quaternion -> Matrix (2 conversions)
        """
        alpha, beta, gamma = sample_euler_angles

        def convert_chain():
            q = euler_to_quaternion(
                alpha, beta, gamma, convention="xyz"
            )
            R = quaternion_to_rotation_matrix(q)
            return R

        result = benchmark(convert_chain)
        assert result.shape == (3, 3)

    def test_matrix_to_axis_angle_chain(
        self, benchmark, sample_rotation_matrix
    ):
        """Benchmark matrix to axis-angle conversion chain.

        SLA: < 50ms
        Tests: Matrix -> Quaternion -> Axis-Angle (2 conversions)
        """
        def convert_chain():
            q = rotation_matrix_to_quaternion(sample_rotation_matrix)
            axis, angle = quaternion_to_axis_angle(q)
            return axis, angle

        axis, angle = benchmark(convert_chain)
        assert len(axis) == 3
        assert isinstance(angle, (float, np.floating))

    def test_quaternion_multiply_sequence(
        self, benchmark, sample_quaternion
    ):
        """Benchmark sequential quaternion multiplications.

        SLA: < 50ms
        Tests: 10 quaternion multiplications
        """
        q1 = sample_quaternion
        q2 = np.array([0.7071, 0.0, 0.7071, 0.0])

        def multiply_sequence():
            result = q1.copy()
            for _ in range(10):
                result = quaternion_multiply(result, q2)
            return result

        result = benchmark(multiply_sequence)
        assert len(result) == 4


@pytest.mark.performance
class TestRotationConverterScaling:
    """Performance benchmarks for scaling with input count."""

    def test_batch_euler_conversions(
        self, benchmark, sample_euler_angles
    ):
        """Benchmark batch conversion of 100 Euler angle sets.

        SLA: < 50ms (amortized per conversion)
        Tests: Throughput with repeated conversions
        """
        alpha, beta, gamma = sample_euler_angles

        def convert_batch():
            results = []
            for i in range(100):
                # Vary angles slightly for realism
                q = euler_to_quaternion(
                    alpha + i * 0.1,
                    beta + i * 0.1,
                    gamma + i * 0.1,
                    convention="xyz",
                )
                results.append(q)
            return results

        results = benchmark(convert_batch)
        assert len(results) == 100

    def test_normalize_quaternion_sequence(
        self, benchmark, sample_quaternion
    ):
        """Benchmark 50 quaternion normalization operations.

        SLA: < 50ms (amortized)
        Tests: Normalization throughput
        """
        q = sample_quaternion

        def normalize_batch():
            results = []
            for i in range(50):
                # Denormalize slightly and re-normalize
                q_dirty = q * (1.0 + i * 0.001)
                q_clean = normalize_quaternion(q_dirty)
                results.append(q_clean)
            return results

        results = benchmark(normalize_batch)
        assert len(results) == 50
