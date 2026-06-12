"""TDD tests for Rust-compiled tools_core Python bindings.

These tests validate that the PyO3 bridge correctly exposes
the Rust simulation kernel to Python consumers.

Principles:
- TDD: Tests define the expected contract for the Python API.
- DbC: Error handling (ValueError) matches Rust Result<T, E> contract.
- DRY: Tests use the same tools_core import that downstream repos will use.
"""

from __future__ import annotations

import math

import pytest

# This import uses the Rust-compiled wheel built by Maturin
tools_core = pytest.importorskip(
    "tools_core",
    reason="tools_core wheel not installed (run: maturin develop --features python)",
)


class TestVector3Construction:
    """Test Vector3 construction and basic properties."""

    def test_create_vector(self) -> None:
        """Vector3(x, y, z) must create a vector with the given components."""
        v = tools_core.Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_repr(self) -> None:
        """repr() must produce a reproducible string."""
        v = tools_core.Vector3(1.0, 2.0, 3.0)
        r = repr(v)
        assert "Vector3" in r
        assert "1" in r

    def test_str(self) -> None:
        """str() must produce a human-readable format."""
        v = tools_core.Vector3(1.0, 2.0, 3.0)
        s = str(v)
        assert "1.000000" in s


class TestVector3Magnitude:
    """Test magnitude calculations."""

    def test_unit_vector(self) -> None:
        """Unit vector along x must have magnitude 1."""
        v = tools_core.Vector3(1.0, 0.0, 0.0)
        assert abs(v.magnitude() - 1.0) < 1e-12

    def test_3_4_5_vector(self) -> None:
        """Vector (3, 4, 0) must have magnitude 5."""
        v = tools_core.Vector3(3.0, 4.0, 0.0)
        assert abs(v.magnitude() - 5.0) < 1e-12


class TestVector3DotProduct:
    """Test dot product correctness."""

    def test_orthogonal(self) -> None:
        """Orthogonal vectors must have zero dot product."""
        a = tools_core.Vector3(1.0, 0.0, 0.0)
        b = tools_core.Vector3(0.0, 1.0, 0.0)
        assert abs(a.dot(b)) < 1e-12

    def test_parallel(self) -> None:
        """Parallel vectors dot product must equal product of magnitudes."""
        a = tools_core.Vector3(2.0, 0.0, 0.0)
        b = tools_core.Vector3(3.0, 0.0, 0.0)
        assert abs(a.dot(b) - 6.0) < 1e-12


class TestVector3CrossProduct:
    """Test cross product correctness."""

    def test_x_cross_y_is_z(self) -> None:
        """x × y must equal z (right-hand rule)."""
        x = tools_core.Vector3(1.0, 0.0, 0.0)
        y = tools_core.Vector3(0.0, 1.0, 0.0)
        z = x.cross(y)
        assert abs(z.x) < 1e-12
        assert abs(z.y) < 1e-12
        assert abs(z.z - 1.0) < 1e-12


class TestVector3Normalization:
    """Test normalization including DbC error handling."""

    def test_normalize_nonzero(self) -> None:
        """Normalized vector must have magnitude 1."""
        v = tools_core.Vector3(3.0, 4.0, 0.0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 1e-12

    def test_normalize_zero_raises(self) -> None:
        """Normalizing zero vector must raise ValueError (DbC contract)."""
        v = tools_core.Vector3(0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="zero"):
            v.normalized()


class TestMathFunctions:
    """Test module-level math functions."""

    def test_lerp_midpoint(self) -> None:
        """lerp(0, 100, 0.5) must return 50."""
        assert abs(tools_core.lerp(0.0, 100.0, 0.5) - 50.0) < 1e-12

    def test_lerp_endpoints(self) -> None:
        """lerp at t=0 and t=1 must return the endpoints."""
        assert abs(tools_core.lerp(10.0, 20.0, 0.0) - 10.0) < 1e-12
        assert abs(tools_core.lerp(10.0, 20.0, 1.0) - 20.0) < 1e-12

    def test_clamp_within_range(self) -> None:
        """Value within range must be unchanged."""
        assert abs(tools_core.clamp(5.0, 0.0, 10.0) - 5.0) < 1e-12

    def test_clamp_below(self) -> None:
        """Value below min must be clamped to min."""
        assert abs(tools_core.clamp(-5.0, 0.0, 10.0) - 0.0) < 1e-12

    def test_clamp_above(self) -> None:
        """Value above max must be clamped to max."""
        assert abs(tools_core.clamp(15.0, 0.0, 10.0) - 10.0) < 1e-12


class TestPythonRustParity:
    """Verify that Rust results match pure Python math exactly (DRY parity)."""

    def test_magnitude_matches_python_math(self) -> None:
        """Rust magnitude must match Python's math.sqrt calculation."""
        x, y, z = 3.0, 4.0, 5.0
        v = tools_core.Vector3(x, y, z)
        python_mag = math.sqrt(x * x + y * y + z * z)
        assert abs(v.magnitude() - python_mag) < 1e-12

    def test_dot_matches_python(self) -> None:
        """Rust dot product must match Python manual calculation."""
        a = tools_core.Vector3(1.0, 2.0, 3.0)
        b = tools_core.Vector3(4.0, 5.0, 6.0)
        python_dot = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0
        assert abs(a.dot(b) - python_dot) < 1e-12
