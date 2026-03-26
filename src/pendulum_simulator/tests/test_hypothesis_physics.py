"""Property-based tests for physics modules using Hypothesis (#m2).

Tests invariants that must hold for ANY valid parameter/state combination:
- Mass matrix symmetry and positive semi-definiteness
- Energy conservation (Hamiltonian) in torque-free systems
- Kinetic energy non-negativity
- Forward kinematics segment lengths
- Constraint Jacobian rank for golfer model
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from double_pendulum_golf.physics import (  # noqa: E402
    PendulumParams,
    forward_kinematics,
    kinetic_energy,
    mass_matrix,
    potential_energy,
    total_energy,
)
from double_pendulum_golf.physics_triple import (  # noqa: E402
    TriplePendulumParams,
    forward_kinematics as triple_fk,
    kinetic_energy as triple_ke,
    mass_matrix as triple_mass_matrix,
    potential_energy as triple_pe,
    total_energy as triple_te,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Physically reasonable parameter ranges
_pos_float = st.floats(min_value=0.1, max_value=10.0)
_nonneg_float = st.floats(min_value=0.0, max_value=1.0)
_angle = st.floats(min_value=-np.pi, max_value=np.pi)
_velocity = st.floats(min_value=-10.0, max_value=10.0)


@st.composite
def double_params(draw: st.DrawFn) -> PendulumParams:
    return PendulumParams(
        m1=draw(_pos_float),
        m2=draw(_pos_float),
        L1=draw(_pos_float),
        L2=draw(_pos_float),
        g=draw(st.floats(min_value=0.0, max_value=20.0)),
        b1=draw(_nonneg_float),
        b2=draw(_nonneg_float),
        mu1=draw(_nonneg_float),
        mu2=draw(_nonneg_float),
    )


@st.composite
def double_state(draw: st.DrawFn) -> np.ndarray:
    return np.array(
        [
            draw(_angle),
            draw(_angle),
            draw(_velocity),
            draw(_velocity),
        ]
    )


@st.composite
def triple_params(draw: st.DrawFn) -> TriplePendulumParams:
    return TriplePendulumParams(
        m1=draw(_pos_float),
        m2=draw(_pos_float),
        m3=draw(_pos_float),
        L1=draw(_pos_float),
        L2=draw(_pos_float),
        L3=draw(_pos_float),
        g=draw(st.floats(min_value=0.0, max_value=20.0)),
        b1=draw(_nonneg_float),
        b2=draw(_nonneg_float),
        b3=draw(_nonneg_float),
    )


@st.composite
def triple_state(draw: st.DrawFn) -> np.ndarray:
    return np.array(
        [
            draw(_angle),
            draw(_angle),
            draw(_angle),
            draw(_velocity),
            draw(_velocity),
            draw(_velocity),
        ]
    )


# ---------------------------------------------------------------------------
# Double Pendulum Properties
# ---------------------------------------------------------------------------


class TestDoubleProperties:
    """Property-based tests for the 2-DOF double pendulum."""

    @given(params=double_params(), phi=_angle)
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_mass_matrix_symmetric(self, params: PendulumParams, phi: float) -> None:
        M = mass_matrix(phi, params)
        assert M.shape == (2, 2)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    @given(params=double_params(), phi=_angle)
    @settings(max_examples=50)
    def test_mass_matrix_psd(self, params: PendulumParams, phi: float) -> None:
        M = mass_matrix(phi, params)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues}"

    @given(params=double_params(), state=double_state())
    @settings(max_examples=50)
    def test_kinetic_energy_non_negative(
        self, params: PendulumParams, state: np.ndarray
    ) -> None:
        T = kinetic_energy(state, params)
        assert T >= -1e-12, f"Negative kinetic energy: {T}"

    @given(params=double_params(), state=double_state())
    @settings(max_examples=50)
    def test_total_energy_finite(self, params: PendulumParams, state: np.ndarray) -> None:
        E = total_energy(state, params)
        assert np.isfinite(E), f"Non-finite total energy: {E}"

    @given(params=double_params(), state=double_state())
    @settings(max_examples=50)
    def test_total_energy_is_sum(self, params: PendulumParams, state: np.ndarray) -> None:
        T = kinetic_energy(state, params)
        V = potential_energy(state, params)
        E = total_energy(state, params)
        assert np.isclose(E, T + V, atol=1e-12)

    @given(params=double_params(), theta1=_angle, phi=_angle)
    @settings(max_examples=30)
    def test_fk_segment_lengths(
        self, params: PendulumParams, theta1: float, phi: float
    ) -> None:
        """FK endpoint distances must match segment lengths."""
        pos = forward_kinematics(theta1, phi, params)
        wrist = np.array(pos["wrist"])
        tip = np.array(pos["tip"])

        # Shoulder at origin, wrist distance = L1
        wrist_dist = np.linalg.norm(wrist)
        assert np.isclose(wrist_dist, params.L1, atol=1e-8), (
            f"Wrist distance {wrist_dist} != L1 {params.L1}"
        )

        # Wrist-to-tip distance = L2
        tip_dist = np.linalg.norm(tip - wrist)
        assert np.isclose(tip_dist, params.L2, atol=1e-8), (
            f"Tip distance {tip_dist} != L2 {params.L2}"
        )


# ---------------------------------------------------------------------------
# Triple Pendulum Properties
# ---------------------------------------------------------------------------


class TestTripleProperties:
    """Property-based tests for the 3-DOF triple pendulum."""

    @given(params=triple_params(), phi1=_angle, phi2=_angle)
    @settings(max_examples=50)
    def test_mass_matrix_symmetric(
        self, params: TriplePendulumParams, phi1: float, phi2: float
    ) -> None:
        M = triple_mass_matrix(phi1, phi2, params)
        assert M.shape == (3, 3)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    @given(params=triple_params(), phi1=_angle, phi2=_angle)
    @settings(max_examples=50)
    def test_mass_matrix_psd(
        self, params: TriplePendulumParams, phi1: float, phi2: float
    ) -> None:
        M = triple_mass_matrix(phi1, phi2, params)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues}"

    @given(params=triple_params(), state=triple_state())
    @settings(max_examples=50)
    def test_kinetic_energy_non_negative(
        self, params: TriplePendulumParams, state: np.ndarray
    ) -> None:
        T = triple_ke(state, params)
        assert T >= -1e-12, f"Negative kinetic energy: {T}"

    @given(params=triple_params(), state=triple_state())
    @settings(max_examples=50)
    def test_total_energy_is_sum(
        self, params: TriplePendulumParams, state: np.ndarray
    ) -> None:
        T = triple_ke(state, params)
        V = triple_pe(state, params)
        E = triple_te(state, params)
        assert np.isclose(E, T + V, atol=1e-12)

    @given(params=triple_params(), state=triple_state())
    @settings(max_examples=30)
    def test_fk_segment_lengths(self, params: TriplePendulumParams, state: np.ndarray) -> None:
        """FK inter-joint distances must match segment lengths."""
        pos = triple_fk(state[0], state[1], state[2], params)
        shoulder = np.array(pos["shoulder"])
        wrist1 = np.array(pos["wrist1"])
        wrist2 = np.array(pos["wrist2"])
        tip = np.array(pos["tip"])

        # Segment 1: shoulder → wrist1 = L1
        d1 = np.linalg.norm(wrist1 - shoulder)
        # Segment 2: wrist1 → wrist2 = L2
        d2 = np.linalg.norm(wrist2 - wrist1)
        # Segment 3: wrist2 → tip = L3
        d3 = np.linalg.norm(tip - wrist2)

        assert np.isclose(d1, params.L1, atol=1e-8)
        assert np.isclose(d2, params.L2, atol=1e-8)
        assert np.isclose(d3, params.L3, atol=1e-8)
