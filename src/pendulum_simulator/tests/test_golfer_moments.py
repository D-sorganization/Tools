# ruff: noqa: E501
"""Tests for golfer_pendulum_moments in joint_moments module.

TDD: These tests define expected behavior for golfer joint moments.
They will fail if golfer_pendulum_moments is not implemented correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.joint_moments import golfer_pendulum_moments


class TestGolferPendulumMoments:
    """Integration tests for golfer 7-DOF joint moments."""

    @pytest.fixture
    def sample_positions(self) -> dict:
        return {
            "hub": (0.0, -0.5),
            "rs": (0.15, -0.5),
            "re": (0.15, -0.8),
            "rh": (0.15, -1.1),
            "ls": (-0.15, -0.5),
            "le": (-0.15, -0.8),
            "lh": (-0.15, -1.1),
            "club_tip": (0.0, -2.1),
        }

    @pytest.fixture
    def sample_forces(self) -> dict:
        return {
            "hub": (1.0, -9.81),
            "rs": (0.5, -5.0),
            "re": (0.3, -3.0),
            "rh": (0.1, -1.0),
            "ls": (0.5, -5.0),
            "le": (0.3, -3.0),
            "lh": (0.1, -1.0),
        }

    def test_returns_all_21_keys(self, sample_positions, sample_forces):
        """Golfer has 7 joints × 3 quantities = 21 keys."""
        torques = (10.0, 8.0, 6.0, 4.0, 8.0, 6.0, 4.0)
        result = golfer_pendulum_moments(sample_positions, sample_forces, torques, None)
        joints = ["hub", "rs", "re", "rh", "ls", "le", "lh"]
        suffixes = ["_applied_torque", "_moment_of_force", "_total_moment"]
        expected_keys = {f"{j}{s}" for j in joints for s in suffixes}
        assert set(result.keys()) == expected_keys

    def test_applied_torques_preserved(self, sample_positions, sample_forces):
        """Applied torques should be passed through unchanged."""
        torques = (10.0, 8.0, 6.0, 4.0, 8.0, 6.0, 4.0)
        result = golfer_pendulum_moments(sample_positions, sample_forces, torques, None)
        assert result["hub_applied_torque"] == pytest.approx(10.0)
        assert result["rs_applied_torque"] == pytest.approx(8.0)
        assert result["rh_applied_torque"] == pytest.approx(4.0)
        assert result["lh_applied_torque"] == pytest.approx(4.0)

    def test_all_values_finite(self, sample_positions, sample_forces):
        """All computed moments must be finite."""
        torques = (10.0, 8.0, 6.0, 4.0, 8.0, 6.0, 4.0)
        result = golfer_pendulum_moments(sample_positions, sample_forces, torques, None)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is non-finite: {val}"

    def test_total_equals_applied_plus_moment(self, sample_positions, sample_forces):
        """total = applied + moment_of_force for each joint."""
        torques = (10.0, 8.0, 6.0, 4.0, 8.0, 6.0, 4.0)
        result = golfer_pendulum_moments(sample_positions, sample_forces, torques, None)
        for jname in ["hub", "rs", "re", "rh", "ls", "le", "lh"]:
            applied = result[f"{jname}_applied_torque"]
            moment = result[f"{jname}_moment_of_force"]
            total = result[f"{jname}_total_moment"]
            assert total == pytest.approx(applied + moment), (
                f"{jname}: total {total} != applied {applied} + moment {moment}"
            )

    def test_too_few_torques_raises(self, sample_positions, sample_forces):
        """Must have at least 7 applied torques."""
        with pytest.raises((ValueError, TypeError), match="Need >= 7"):
            golfer_pendulum_moments(sample_positions, sample_forces, (1.0, 2.0), None)

    def test_zero_torques_yield_nonzero_moments(self, sample_positions, sample_forces):
        """With nonzero forces, moments of force should be nonzero even with zero applied."""
        torques = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        result = golfer_pendulum_moments(sample_positions, sample_forces, torques, None)
        # At least some moments should be nonzero (forces are not parallel to lever arms)
        moment_values = [result[f"{j}_moment_of_force"] for j in ["hub", "rs", "re"]]
        assert any(abs(m) > 1e-6 for m in moment_values)

    def test_missing_position_key_graceful(self, sample_forces):
        """Missing position key should produce zeros for that joint, not crash."""
        positions = {
            "hub": (0.0, -0.5),
            "rs": (0.15, -0.5),
            # re, rh, ls, le, lh, club_tip missing
        }
        torques = (10.0, 8.0, 6.0, 4.0, 8.0, 6.0, 4.0)
        result = golfer_pendulum_moments(positions, sample_forces, torques, None)
        # Joints with missing data should still have entries
        assert "re_applied_torque" in result
        assert result["re_applied_torque"] == pytest.approx(6.0)
