"""Tests for the data extractor module.

Covers:
- list_available_series() public API
- extract_series() for all registered series
- Error handling (unknown keys)
- Factory functions are closed over correctly
- Contract: returned array is 1-D and same length as trajectory
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from double_pendulum_golf.data_extractor import (
    _DOUBLE_SERIES,
    extract_series,
    list_available_series,
)

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_mock_result(n: int = 5) -> MagicMock:
    """Build a minimal mock SimulationResult supporting all extractor interfaces."""
    r = MagicMock()
    r.n_steps = n
    # Trajectory arrays
    r.t = np.linspace(0.0, 1.0, n)
    r.states = np.random.default_rng(0).random((n, 4))

    # torques_at(i) → array-like length 2
    r.torques_at.side_effect = lambda i: np.array([float(i), float(i) * 2])

    # all_total_torques() → (n, 2)
    r.all_total_torques.return_value = np.ones((n, 2))

    # all_energies() → dict with 'kinetic', 'potential', 'total'
    r.all_energies.return_value = {
        "kinetic": np.linspace(1, 2, n),
        "potential": np.linspace(0, 1, n),
        "total": np.linspace(1, 3, n),
    }

    # joint_velocities_at(i) → dict with 'wrist_speed', 'tip_speed'
    r.joint_velocities_at.side_effect = lambda i: {
        "wrist_speed": float(i) * 0.5,
        "tip_speed": float(i) * 1.0,
    }

    # all_accelerations() → (n, 2)
    r.all_accelerations.return_value = np.ones((n, 2)) * 0.1

    # coriolis_at(i) → array length 2
    r.coriolis_at.side_effect = lambda i: np.array([float(i) * 0.01, float(i) * 0.02])

    # gravity_at(i) → array length 2
    r.gravity_at.side_effect = lambda i: np.array([9.81, 0.0])

    # all_friction_torques() → (n, 2)
    r.all_friction_torques.return_value = np.zeros((n, 2))

    # base_force_at(i) → dict with 'fx', 'fy', 'magnitude'
    r.base_force_at.side_effect = lambda i: {
        "fx": float(i),
        "fy": float(i) * 0.5,
        "magnitude": float(i) * 1.12,
    }

    return r


# ---------------------------------------------------------------------------
# Tests for list_available_series
# ---------------------------------------------------------------------------


class TestListAvailableSeries:
    def test_returns_list(self) -> None:
        result = list_available_series()
        assert isinstance(result, list)

    def test_returns_all_double_series(self) -> None:
        result = list_available_series()
        keys = [k for k, _, _ in result]
        assert set(keys) == set(_DOUBLE_SERIES.keys())

    def test_each_item_is_three_tuple(self) -> None:
        for item in list_available_series():
            assert isinstance(item, tuple)
            assert len(item) == 3
            key, desc, unit = item
            assert isinstance(key, str) and len(key) > 0
            assert isinstance(desc, str) and len(desc) > 0
            assert isinstance(unit, str) and len(unit) > 0

    def test_time_series_present(self) -> None:
        keys = [k for k, _, _ in list_available_series()]
        assert "time" in keys

    def test_torque_series_present(self) -> None:
        keys = [k for k, _, _ in list_available_series()]
        assert "torque_shoulder" in keys
        assert "torque_wrist" in keys

    def test_energy_series_present(self) -> None:
        keys = [k for k, _, _ in list_available_series()]
        assert "kinetic_energy" in keys
        assert "potential_energy" in keys
        assert "total_energy" in keys

    def test_model_type_default_is_double(self) -> None:
        default = list_available_series()
        explicit = list_available_series("double")
        assert default == explicit

    def test_non_double_model_type_still_returns_double(self) -> None:
        # Currently the registry only has double series; other model_type strings
        # fall through to the same registry.
        result = list_available_series("triple")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests for extract_series
# ---------------------------------------------------------------------------


class TestExtractSeries:
    @pytest.fixture
    def mock_result(self) -> MagicMock:
        return _make_mock_result(n=6)

    def test_raises_on_unknown_key(self, mock_result: MagicMock) -> None:
        with pytest.raises(KeyError, match="unknown_xyz"):
            extract_series(mock_result, "unknown_xyz")

    def test_returns_tuple_of_three(self, mock_result: MagicMock) -> None:
        values, desc, unit = extract_series(mock_result, "time")
        assert isinstance(values, np.ndarray)
        assert isinstance(desc, str)
        assert isinstance(unit, str)

    def test_time_series(self, mock_result: MagicMock) -> None:
        values, desc, unit = extract_series(mock_result, "time")
        assert values.ndim == 1
        assert len(values) == 6
        np.testing.assert_allclose(values, mock_result.t)

    def test_theta1_series(self, mock_result: MagicMock) -> None:
        values, desc, unit = extract_series(mock_result, "theta1")
        assert values.ndim == 1
        assert len(values) == 6
        np.testing.assert_allclose(values, mock_result.states[:, 0])

    def test_phi_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "phi")
        np.testing.assert_allclose(values, mock_result.states[:, 1])

    def test_dtheta1_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "dtheta1")
        np.testing.assert_allclose(values, mock_result.states[:, 2])

    def test_dphi_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "dphi")
        np.testing.assert_allclose(values, mock_result.states[:, 3])

    def test_torque_shoulder_series(self, mock_result: MagicMock) -> None:
        values, desc, unit = extract_series(mock_result, "torque_shoulder")
        assert values.ndim == 1
        assert len(values) == 6
        assert unit == "N·m"

    def test_torque_wrist_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "torque_wrist")
        assert values.ndim == 1
        assert len(values) == 6

    def test_total_torque_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "total_torque_shoulder")
        assert values.ndim == 1
        total = mock_result.all_total_torques()
        np.testing.assert_allclose(values, total[:, 0])

    def test_kinetic_energy_series(self, mock_result: MagicMock) -> None:
        values, _, unit = extract_series(mock_result, "kinetic_energy")
        assert values.ndim == 1
        energies = mock_result.all_energies()
        np.testing.assert_allclose(values, energies["kinetic"])
        assert unit == "J"

    def test_potential_energy_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "potential_energy")
        energies = mock_result.all_energies()
        np.testing.assert_allclose(values, energies["potential"])

    def test_total_energy_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "total_energy")
        energies = mock_result.all_energies()
        np.testing.assert_allclose(values, energies["total"])

    def test_wrist_speed_series(self, mock_result: MagicMock) -> None:
        values, _, unit = extract_series(mock_result, "wrist_speed")
        assert values.ndim == 1
        assert unit == "m/s"
        # Values come from joint_velocities_at(i)["wrist_speed"] = i * 0.5
        expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        np.testing.assert_allclose(values, expected)

    def test_tip_speed_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "tip_speed")
        assert values.ndim == 1

    def test_accel_shoulder_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "accel_shoulder")
        assert values.ndim == 1
        accels = mock_result.all_accelerations()
        np.testing.assert_allclose(values, accels[:, 0])

    def test_accel_wrist_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "accel_wrist")
        accels = mock_result.all_accelerations()
        np.testing.assert_allclose(values, accels[:, 1])

    def test_coriolis_shoulder_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "coriolis_shoulder")
        assert values.ndim == 1
        # coriolis_at(i)[0] = i * 0.01
        expected = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
        np.testing.assert_allclose(values, expected)

    def test_coriolis_wrist_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "coriolis_wrist")
        assert values.ndim == 1

    def test_gravity_shoulder_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "gravity_shoulder")
        assert values.ndim == 1
        # gravity_at(i)[0] = 9.81 for all i
        np.testing.assert_allclose(values, np.full(6, 9.81))

    def test_gravity_wrist_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "gravity_wrist")
        np.testing.assert_allclose(values, np.zeros(6))

    def test_friction_shoulder_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "friction_shoulder")
        friction = mock_result.all_friction_torques()
        np.testing.assert_allclose(values, friction[:, 0])

    def test_friction_wrist_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "friction_wrist")
        friction = mock_result.all_friction_torques()
        np.testing.assert_allclose(values, friction[:, 1])

    def test_base_force_x_series(self, mock_result: MagicMock) -> None:
        values, _, unit = extract_series(mock_result, "base_force_x")
        assert values.ndim == 1
        assert unit == "N"
        expected = np.arange(6, dtype=float)
        np.testing.assert_allclose(values, expected)

    def test_base_force_y_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "base_force_y")
        assert values.ndim == 1

    def test_base_force_magnitude_series(self, mock_result: MagicMock) -> None:
        values, _, _ = extract_series(mock_result, "base_force_mag")
        assert values.ndim == 1

    def test_all_registered_keys_extractable(self, mock_result: MagicMock) -> None:
        """Smoke-test: every key in _DOUBLE_SERIES can be extracted."""
        for key in _DOUBLE_SERIES:
            values, desc, unit = extract_series(mock_result, key)
            assert isinstance(values, np.ndarray)
            assert values.ndim == 1
            assert len(desc) > 0
            assert len(unit) > 0

    def test_extract_series_postcondition_1d(self, mock_result: MagicMock) -> None:
        """Contract: returned arrays must always be 1-D."""
        for key in _DOUBLE_SERIES:
            values, _, _ = extract_series(mock_result, key)
            assert values.ndim == 1, f"Key '{key}' returned {values.ndim}-D array"
