"""Unit tests for the SwingSource protocol and DoublePendulumSwing."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.swing_source import DoublePendulumSwing, SwingSource
from shared.python.swing_sim.types import (
    PendulumState,
    PlaneOrientation,
    SwingSample,
)


def _small_swing(**kwargs: object) -> DoublePendulumSwing:
    defaults: dict[str, object] = {
        "duration": 0.1,
        "dt": 1e-3,
        "backend": "python",
    }
    defaults.update(kwargs)
    return DoublePendulumSwing(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestDoublePendulumSwing:
    def test_conforms_to_swing_source_protocol(self) -> None:
        swing = _small_swing()
        assert isinstance(swing, SwingSource)

    def test_duration_matches_grid(self) -> None:
        swing = _small_swing(duration=0.1, dt=1e-3)
        assert swing.duration == pytest.approx(0.1)

    def test_python_backend_selected_when_forced(self) -> None:
        assert _small_swing(backend="python").backend == "python"

    def test_sample_returns_valid_swing_sample(self) -> None:
        swing = _small_swing()
        sample = swing.sample(0.05)
        assert isinstance(sample, SwingSample)
        assert sample.t == pytest.approx(0.05)
        rotation = sample.pose[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        assert np.all(np.isfinite(sample.twist))

    def test_sample_at_zero_matches_initial_geometry(self) -> None:
        initial = PendulumState(theta1=np.pi / 2.0, theta2=0.0, omega1=0.0, omega2=0.0)
        swing = _small_swing(initial_state=initial, plane=PlaneOrientation())
        sample = swing.sample(0.0)
        p = swing.parameters
        # theta1 = pi/2, theta2 = 0: both segments horizontal along +x.
        expected = np.array([p.l1 + p.l2, 0.0, 0.0])
        np.testing.assert_allclose(sample.pose[:3, 3], expected, atol=1e-9)
        # At rest: zero twist.
        np.testing.assert_allclose(sample.twist, np.zeros(6), atol=1e-12)

    def test_flat_plane_motion_stays_in_plane(self) -> None:
        swing = _small_swing(plane=PlaneOrientation())
        for t in (0.0, 0.03, 0.07, 0.1):
            sample = swing.sample(t)
            # Identity plane spans world x/z; normal is world y.
            assert sample.pose[1, 3] == pytest.approx(0.0, abs=1e-12)

    def test_yawed_plane_rotates_positions(self) -> None:
        flat = _small_swing(plane=PlaneOrientation())
        yawed = _small_swing(plane=PlaneOrientation(yaw_deg=90.0))
        p_flat = flat.sample(0.05).pose[:3, 3]
        p_yawed = yawed.sample(0.05).pose[:3, 3]
        # Same in-plane dynamics (gravity yaw-invariant), rotated 90° about z.
        rz90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(p_yawed, rz90 @ p_flat, atol=1e-9)

    def test_sample_outside_duration_raises(self) -> None:
        swing = _small_swing()
        with pytest.raises(ValueError, match="duration"):
            swing.sample(swing.duration + 0.1)
        with pytest.raises(ValueError, match="duration"):
            swing.sample(-0.1)

    def test_rejects_nonpositive_duration(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            _small_swing(duration=0.0)

    def test_rejects_dt_exceeding_duration(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            _small_swing(duration=0.01, dt=0.1)

    def test_rust_backend_strict_posture(self) -> None:
        from shared.python.swing_sim import _rust_facade

        if _rust_facade.rust_available():
            # Wheel present: rust backend must work and be reported.
            assert _small_swing(backend="rust").backend == "rust"
        else:
            # Wheel absent: forcing rust must raise, never silently degrade.
            with pytest.raises(ImportError, match="swing_core"):
                _small_swing(backend="rust")
