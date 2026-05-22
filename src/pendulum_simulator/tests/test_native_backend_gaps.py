"""Gap-fill tests for native_backend.py — covers remaining uncovered lines.

Line 32:   _NATIVE_IMPORT_ERROR = None (only when native IS available)
Lines 150, 167, 186: _to_rust_* raises RuntimeError when pendulum_core is None

These are exercised by directly calling the private functions.
"""

from __future__ import annotations

import pytest

from double_pendulum_golf import native_backend
from double_pendulum_golf.native_backend import golfer_native_available
from double_pendulum_golf.physics import PendulumParams
from double_pendulum_golf.physics_golfer import GolferParams
from double_pendulum_golf.physics_triple import TriplePendulumParams


@pytest.fixture
def double_params() -> PendulumParams:
    return PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)


@pytest.fixture
def triple_params() -> TriplePendulumParams:
    return TriplePendulumParams(m1=5.0, m2=3.0, m3=0.5, L1=0.6, L2=0.6, L3=0.6)


@pytest.fixture
def golfer_params() -> GolferParams:
    return GolferParams(
        m_hub=2.0,
        m_r_upper=3.0,
        m_r_fore=2.0,
        m_l_upper=3.0,
        m_l_fore=2.0,
        m_club=0.5,
        L_hub=0.15,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.10,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


# ===========================================================================
# _to_rust_* functions raise RuntimeError when pendulum_core is None
# ===========================================================================


@pytest.mark.skipif(
    golfer_native_available(),
    reason="Only tests the unavailable path",
)
class TestToRustFunctionsWhenUnavailable:
    """These functions raise RuntimeError when _pendulum_core is None."""

    def test_to_rust_double_raises(self, double_params: PendulumParams) -> None:
        with pytest.raises(RuntimeError):
            native_backend._to_rust_double_params(double_params)

    def test_to_rust_triple_raises(self, triple_params: TriplePendulumParams) -> None:
        with pytest.raises(RuntimeError):
            native_backend._to_rust_triple_params(triple_params)

    def test_to_rust_golfer_raises(self, golfer_params: GolferParams) -> None:
        with pytest.raises(RuntimeError):
            native_backend._to_rust_golfer_params(golfer_params)


# ===========================================================================
# _NATIVE_IMPORT_ERROR reflects actual import state
# ===========================================================================


class TestNativeImportError:
    def test_import_error_is_none_or_string(self) -> None:
        err = native_backend._NATIVE_IMPORT_ERROR
        assert err is None or isinstance(err, str)

    def test_import_error_consistent_with_availability(self) -> None:
        """If native is available, import error should be None."""
        if golfer_native_available():
            assert native_backend._NATIVE_IMPORT_ERROR is None
        else:
            assert native_backend._NATIVE_IMPORT_ERROR is not None
            assert isinstance(native_backend._NATIVE_IMPORT_ERROR, str)

    def test_get_native_backend_info_includes_error(self) -> None:
        from double_pendulum_golf.native_backend import get_native_backend_info

        info = get_native_backend_info()
        assert "native_import_error" in info
        if golfer_native_available():
            assert info["native_import_error"] is None
        else:
            assert info["native_import_error"] is not None


# ===========================================================================
# golfer_native_constraint_dynamics_supported — fringe coverage
# ===========================================================================


class TestGolferNativeConstraintDynamicsSupported:
    def test_with_zero_b_returns_true(self, golfer_params: GolferParams) -> None:
        """Default params all have zero b — should return True."""
        from double_pendulum_golf.native_backend import (
            golfer_native_constraint_dynamics_supported,
        )

        assert golfer_native_constraint_dynamics_supported(golfer_params) is True

    def test_with_nonzero_b_hub_returns_false(self, golfer_params: GolferParams) -> None:
        from double_pendulum_golf.native_backend import (
            golfer_native_constraint_dynamics_supported,
        )
        import dataclasses

        p_with_b = dataclasses.replace(golfer_params, b_hub=0.5)
        assert golfer_native_constraint_dynamics_supported(p_with_b) is False
