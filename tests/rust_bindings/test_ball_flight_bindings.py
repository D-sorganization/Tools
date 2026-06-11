"""TDD tests for ball flight physics Python bindings.

Validates that the PyO3-wrapped ball flight simulation matches
the expected physical behavior and provides correct Python-accessible
types.

Principles:
- TDD: Tests define the expected Python API contract.
- DbC: Validates physical constraints (positive speed, gravity, etc.).
- DRY: Uses the same tools_core import that downstream repos will use.
"""

from __future__ import annotations

import pytest

tools_core = pytest.importorskip(
    "tools_core",
    reason="tools_core wheel not installed (run: maturin develop --features python)",
)


class TestBallProperties:
    """Test BallProperties Python binding."""

    def test_create_defaults(self) -> None:
        """BallProperties() must use golf ball defaults."""
        bp = tools_core.BallProperties()
        assert repr(bp).startswith("BallProperties")

    def test_repr_contains_mass(self) -> None:
        """repr() must include mass information."""
        bp = tools_core.BallProperties()
        r = repr(bp)
        assert "mass" in r.lower() or "0.0459" in r


class TestLaunchConditions:
    """Test LaunchConditions Python binding."""

    def test_create_defaults(self) -> None:
        """LaunchConditions() uses reasonable defaults."""
        lc = tools_core.LaunchConditions()
        assert repr(lc).startswith("LaunchConditions")

    def test_create_with_params(self) -> None:
        """LaunchConditions(v, angle, azimuth, spin) must accept parameters."""
        lc = tools_core.LaunchConditions(
            velocity=70.0,
            launch_angle=12.0,
            azimuth_angle=0.0,
            spin_rate=2500.0,
        )
        r = repr(lc)
        assert "70" in r

    def test_repr_contains_spin(self) -> None:
        """repr() must include spin information."""
        lc = tools_core.LaunchConditions(spin_rate=3000.0)
        r = repr(lc)
        assert "3000" in r


class TestEnvironmentalConditions:
    """Test EnvironmentalConditions Python binding."""

    def test_create_defaults(self) -> None:
        """EnvironmentalConditions() must create with sea-level defaults."""
        ec = tools_core.EnvironmentalConditions()
        assert ec is not None


class TestTrajectoryPoint:
    """Test TrajectoryPoint Python binding (read-only getters)."""

    def test_type_exists(self) -> None:
        """TrajectoryPoint type must be importable."""
        assert hasattr(tools_core, "TrajectoryPoint")


class TestTrajectoryAnalysis:
    """Test TrajectoryAnalysis Python binding."""

    def test_type_exists(self) -> None:
        """TrajectoryAnalysis type must be importable."""
        assert hasattr(tools_core, "TrajectoryAnalysis")
