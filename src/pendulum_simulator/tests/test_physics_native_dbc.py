"""Tests for DbC preconditions in physics_native module (GH1478).

Validates that DoublePendulumParams, GolferParams, DoublePendulum, and Golfer
constructors and methods enforce preconditions with TypeError/ValueError.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add the pendulum-core python directory to path so physics_native is importable.
# This sys.path manipulation must precede the physics_native import.
_NATIVE_DIR = str(Path(__file__).resolve().parent.parent / "pendulum-core" / "python")
if _NATIVE_DIR not in sys.path:
    sys.path.insert(0, _NATIVE_DIR)

from physics_native import DoublePendulum, DoublePendulumParams  # noqa: E402

# ---------------------------------------------------------------------------
# DoublePendulumParams constructor preconditions
# ---------------------------------------------------------------------------


class TestDoublePendulumParamsDbc:
    """DbC preconditions on DoublePendulumParams.__init__."""

    def test_m1_must_be_positive(self):
        with pytest.raises(ValueError, match="m1 must be positive"):
            DoublePendulumParams(m1=-1.0, m2=1.0, l1=1.0, l2=1.0)

    def test_m1_zero_rejected(self):
        with pytest.raises(ValueError, match="m1 must be positive"):
            DoublePendulumParams(m1=0.0, m2=1.0, l1=1.0, l2=1.0)

    def test_m2_must_be_positive(self):
        with pytest.raises(ValueError, match="m2 must be positive"):
            DoublePendulumParams(m1=1.0, m2=-0.5, l1=1.0, l2=1.0)

    def test_l1_must_be_positive(self):
        with pytest.raises(ValueError, match="l1 must be positive"):
            DoublePendulumParams(m1=1.0, m2=1.0, l1=0.0, l2=1.0)

    def test_l2_must_be_positive(self):
        with pytest.raises(ValueError, match="l2 must be positive"):
            DoublePendulumParams(m1=1.0, m2=1.0, l1=1.0, l2=-2.0)

    def test_g_must_be_positive(self):
        with pytest.raises(ValueError, match="g must be positive"):
            DoublePendulumParams(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=0.0)

    def test_friction_negative_rejected(self):
        with pytest.raises(ValueError, match="friction1 must be non-negative"):
            DoublePendulumParams(m1=1.0, m2=1.0, l1=1.0, l2=1.0, friction1=-0.1)

    def test_friction2_negative_rejected(self):
        with pytest.raises(ValueError, match="friction2 must be non-negative"):
            DoublePendulumParams(m1=1.0, m2=1.0, l1=1.0, l2=1.0, friction2=-0.1)

    def test_m_clubhead_negative_rejected(self):
        with pytest.raises(ValueError, match="m_clubhead must be non-negative"):
            DoublePendulumParams(m1=1.0, m2=1.0, l1=1.0, l2=1.0, m_clubhead=-0.01)

    def test_m1_wrong_type_rejected(self):
        with pytest.raises(TypeError, match="m1 must be a number"):
            DoublePendulumParams(m1="heavy", m2=1.0, l1=1.0, l2=1.0)

    def test_valid_params_accepted(self):
        params = DoublePendulumParams(m1=1.0, m2=0.5, l1=0.8, l2=0.6)
        assert params.m1 == 1.0
        assert params.m2 == 0.5

    def test_zero_friction_accepted(self):
        params = DoublePendulumParams(m1=1.0, m2=1.0, l1=1.0, l2=1.0, friction1=0.0)
        assert params.friction1 == 0.0

    def test_zero_clubhead_accepted(self):
        params = DoublePendulumParams(m1=1.0, m2=1.0, l1=1.0, l2=1.0, m_clubhead=0.0)
        assert params.m_clubhead == 0.0


# ---------------------------------------------------------------------------
# DoublePendulum method preconditions
# ---------------------------------------------------------------------------


class TestDoublePendulumMethodsDbc:
    """DbC preconditions on DoublePendulum methods."""

    @pytest.fixture
    def model(self):
        return DoublePendulum(m1=1.0, m2=0.5, l1=1.0, l2=0.8)

    def test_mass_matrix_wrong_type_rejected(self, model):
        with pytest.raises(TypeError, match="q must be a numpy ndarray"):
            model.mass_matrix([0.0, 0.0])

    def test_mass_matrix_wrong_shape_rejected(self, model):
        with pytest.raises(ValueError, match="q must have shape"):
            model.mass_matrix(np.array([0.0, 0.0, 0.0]))

    def test_gravity_vector_wrong_type_rejected(self, model):
        with pytest.raises(TypeError, match="q must be a numpy ndarray"):
            model.gravity_vector((0.0, 0.0))

    def test_gravity_vector_wrong_shape_rejected(self, model):
        with pytest.raises(ValueError, match="q must have shape"):
            model.gravity_vector(np.zeros(4))

    def test_coriolis_q_wrong_type_rejected(self, model):
        with pytest.raises(TypeError, match="q must be a numpy ndarray"):
            model.coriolis([0.0, 0.0], np.zeros(2))

    def test_coriolis_qdot_wrong_type_rejected(self, model):
        with pytest.raises(TypeError, match="qdot must be a numpy ndarray"):
            model.coriolis(np.zeros(2), [0.0, 0.0])

    def test_coriolis_qdot_wrong_shape_rejected(self, model):
        with pytest.raises(ValueError, match="qdot must have shape"):
            model.coriolis(np.zeros(2), np.zeros(3))

    def test_forward_kinematics_wrong_type_rejected(self, model):
        with pytest.raises(TypeError, match="q must be a numpy ndarray"):
            model.forward_kinematics([0.0, 0.0])

    def test_forward_kinematics_wrong_shape_rejected(self, model):
        with pytest.raises(ValueError, match="q must have shape"):
            model.forward_kinematics(np.zeros(3))

    def test_mass_matrix_valid_input(self, model):
        q = np.array([0.0, 0.0])
        M = model.mass_matrix(q)
        assert M.shape == (2, 2)
        assert np.all(np.isfinite(M))

    def test_gravity_vector_valid_input(self, model):
        q = np.array([0.1, 0.2])
        G = model.gravity_vector(q)
        assert G.shape == (2,)
        assert np.all(np.isfinite(G))

    def test_forward_kinematics_valid_input(self, model):
        q = np.array([0.0, 0.0])
        fk = model.forward_kinematics(q)
        assert "wrist_x" in fk
        assert "club_tip_x" in fk


# ---------------------------------------------------------------------------
# Golfer native-required behavior (issue #3294)
# ---------------------------------------------------------------------------

import physics_native  # noqa: E402

_GOLFER_KWARGS = dict(
    l_hub=0.2,
    m_hub=0.01,
    d_rs=0.1,
    d_ls=0.1,
    l_r_upper=0.3,
    m_r_upper=2.0,
    l_r_fore=0.3,
    m_r_fore=1.5,
    l_l_upper=0.3,
    m_l_upper=2.0,
    l_l_fore=0.3,
    m_l_fore=1.5,
    l_club=1.0,
    m_club=0.3,
    m_clubhead=0.2,
    grip_right=0.5,
    grip_left=0.5,
)


class TestGolferNativeRequired:
    """Golfer must fail fast at construction when the native lib is absent (#3294)."""

    def test_construction_fails_fast_without_native(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(physics_native, "HAS_NATIVE", False)
        monkeypatch.setattr(physics_native, "NATIVE_ERROR", "simulated import error")
        with pytest.raises(RuntimeError) as exc:
            physics_native.Golfer(**_GOLFER_KWARGS)
        # Error must point the user at the build/install path, not a fallback.
        message = str(exc.value)
        assert "native" in message.lower()
        assert "maturin" in message.lower() or "wheel" in message.lower()

    def test_no_stale_1736_reference_in_source(self) -> None:
        """The misleading closed-issue reference must be gone (#3294)."""
        source = Path(physics_native.__file__).read_text(encoding="utf-8")
        assert "#1736" not in source

    def test_no_misleading_fallback_log_for_golfer(self) -> None:
        """The golfer mass_matrix must not log a 'falling back to NumPy' promise."""
        source = Path(physics_native.__file__).read_text(encoding="utf-8")
        # The double-pendulum path legitimately falls back; the golfer path must
        # not claim a fallback that does not exist. Assert the specific stale
        # golfer log string is gone.
        assert "golfer mass_matrix call failed (%s), falling back to NumPy" not in source

    @pytest.mark.skipif(not physics_native.HAS_NATIVE, reason="native pendulum_core not built")
    def test_construction_succeeds_with_native(self) -> None:
        golfer = physics_native.Golfer(**_GOLFER_KWARGS)
        assert golfer.use_native is True
