"""Tests for the optional Rust-backed kernel adapters."""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf import native_backend
from double_pendulum_golf.physics import (
    PendulumParams,
    coriolis_vector as double_coriolis_vector,
    forward_kinematics as double_forward_kinematics,
    gravity_vector as double_gravity_vector,
    mass_matrix as double_mass_matrix,
)
from double_pendulum_golf.physics_golfer import (
    GolferParams,
    forward_kinematics,
    get_native_backend_info,
    gravity_vector,
    mass_matrix,
)
from double_pendulum_golf.physics_triple import (
    TriplePendulumParams,
    coriolis_vector as triple_coriolis_vector,
    forward_kinematics as triple_forward_kinematics,
    gravity_vector as triple_gravity_vector,
    mass_matrix as triple_mass_matrix,
)


@pytest.fixture
def double_params() -> PendulumParams:
    return PendulumParams(m1=5.0, m2=0.5, mClub=0.2, L1=0.6, L2=1.0)


@pytest.fixture
def triple_params() -> TriplePendulumParams:
    return TriplePendulumParams(m1=5.0, m2=0.5, m3=0.2, L1=0.6, L2=0.6, L3=0.6)


@pytest.fixture
def golfer_params() -> GolferParams:
    """Representative golfer parameters for backend dispatch tests."""
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
        L_club=1.1,
        d_rs=0.20,
        d_ls=0.20,
        grip_right=0.05,
        grip_left=0.25,
        m_clubhead=0.2,
    )


def test_native_backend_info_defaults_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_backend, "_pendulum_core", None)
    monkeypatch.delenv("PENDULUM_DOUBLE_BACKEND", raising=False)
    monkeypatch.delenv("PENDULUM_TRIPLE_BACKEND", raising=False)
    monkeypatch.delenv("PENDULUM_GOLFER_BACKEND", raising=False)
    info = get_native_backend_info()

    assert info["configured_backend"] == {
        "double": "python",
        "triple": "python",
        "golfer": "python",
    }
    assert info["supported_models"] == {
        "golfer": True,
        "double": True,
        "triple": True,
    }
    assert info["supports_constraint_dynamics"] is True


def test_double_mass_matrix_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, double_params: PendulumParams
) -> None:
    sentinel = np.eye(2) * 11.0
    monkeypatch.setattr(
        native_backend, "double_mass_matrix", lambda phi, params: sentinel.copy()
    )

    result = double_mass_matrix(0.1, double_params)
    assert np.array_equal(result, sentinel)


def test_double_gravity_vector_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, double_params: PendulumParams
) -> None:
    sentinel = np.array([1.0, 2.0])
    monkeypatch.setattr(
        native_backend,
        "double_gravity_vector",
        lambda theta1, phi, params: sentinel.copy(),
    )

    result = double_gravity_vector(0.2, -0.1, double_params)
    assert np.array_equal(result, sentinel)


def test_double_coriolis_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, double_params: PendulumParams
) -> None:
    sentinel = np.array([-3.0, 4.0])
    monkeypatch.setattr(
        native_backend,
        "double_coriolis_vector",
        lambda phi, dtheta1, dphi, params: sentinel.copy(),
    )

    result = double_coriolis_vector(0.3, 1.0, -2.0, double_params)
    assert np.array_equal(result, sentinel)


def test_double_forward_kinematics_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, double_params: PendulumParams
) -> None:
    sentinel = {
        "shoulder": (0.0, 0.0),
        "wrist": (0.1, -0.6),
        "tip": (0.2, -1.6),
    }
    monkeypatch.setattr(
        native_backend,
        "double_forward_kinematics",
        lambda theta1, phi, params: sentinel.copy(),
    )

    result = double_forward_kinematics(0.0, 0.0, double_params)
    assert result == sentinel


def test_triple_mass_matrix_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, triple_params: TriplePendulumParams
) -> None:
    sentinel = np.eye(3) * 13.0
    monkeypatch.setattr(
        native_backend, "triple_mass_matrix", lambda phi1, phi2, params: sentinel.copy()
    )

    result = triple_mass_matrix(0.1, -0.2, triple_params)
    assert np.array_equal(result, sentinel)


def test_triple_gravity_vector_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, triple_params: TriplePendulumParams
) -> None:
    sentinel = np.array([1.0, 2.0, 3.0])
    monkeypatch.setattr(
        native_backend,
        "triple_gravity_vector",
        lambda theta1, phi1, phi2, params: sentinel.copy(),
    )

    result = triple_gravity_vector(0.2, -0.1, 0.05, triple_params)
    assert np.array_equal(result, sentinel)


def test_triple_coriolis_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, triple_params: TriplePendulumParams
) -> None:
    sentinel = np.array([-3.0, 4.0, -5.0])
    monkeypatch.setattr(
        native_backend,
        "triple_coriolis_vector",
        lambda phi1, phi2, dtheta1, dphi1, dphi2, params: sentinel.copy(),
    )

    result = triple_coriolis_vector(0.3, -0.2, 1.0, -2.0, 0.5, triple_params)
    assert np.array_equal(result, sentinel)


def test_triple_forward_kinematics_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, triple_params: TriplePendulumParams
) -> None:
    sentinel = {
        "shoulder": (0.0, 0.0),
        "wrist1": (0.1, -0.6),
        "wrist2": (0.2, -1.2),
        "tip": (0.3, -1.8),
    }
    monkeypatch.setattr(
        native_backend,
        "triple_forward_kinematics",
        lambda theta1, phi1, phi2, params: sentinel.copy(),
    )

    result = triple_forward_kinematics(0.0, 0.0, 0.0, triple_params)
    assert result == sentinel


def test_mass_matrix_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, golfer_params: GolferParams
) -> None:
    sentinel = np.eye(8) * 7.0
    monkeypatch.setattr(
        native_backend, "golfer_mass_matrix", lambda q, params: sentinel.copy()
    )

    result = mass_matrix(np.zeros(8), golfer_params)
    assert np.array_equal(result, sentinel)


def test_gravity_vector_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, golfer_params: GolferParams
) -> None:
    sentinel = np.arange(8, dtype=float)
    monkeypatch.setattr(
        native_backend, "golfer_gravity_vector", lambda q, params: sentinel.copy()
    )

    result = gravity_vector(np.zeros(8), golfer_params)
    assert np.array_equal(result, sentinel)


def test_forward_kinematics_prefers_native_backend(
    monkeypatch: pytest.MonkeyPatch, golfer_params: GolferParams
) -> None:
    sentinel = {
        "origin": (0.0, 0.0),
        "hub": (0.1, -0.2),
        "rs": (0.2, -0.2),
        "re": (0.3, -0.3),
        "rh": (0.4, -0.4),
        "ls": (-0.2, -0.2),
        "le": (-0.3, -0.3),
        "lh": (-0.4, -0.4),
        "club_base": (0.35, -0.35),
        "club_tip": (0.5, -1.0),
        "grip_right": (0.4, -0.4),
        "grip_left": (0.45, -0.8),
    }
    monkeypatch.setattr(
        native_backend,
        "golfer_forward_kinematics",
        lambda q, params: sentinel.copy(),
    )

    result = forward_kinematics(np.zeros(8), golfer_params)
    assert result == sentinel


def test_native_forward_kinematics_maps_rust_keys(
    monkeypatch: pytest.MonkeyPatch, golfer_params: GolferParams
) -> None:
    class StubCore:
        @staticmethod
        def PyGolferParams(*args: float) -> tuple[float, ...]:
            return args

        @staticmethod
        def py_golfer_forward_kinematics(
            q: list[float], params: tuple[float, ...]
        ) -> dict[str, list[float]]:
            del q, params
            return {
                "hub": [0.1, -0.2],
                "r_shoulder": [0.2, -0.2],
                "r_elbow": [0.3, -0.3],
                "r_wrist": [0.4, -0.4],
                "l_shoulder": [-0.2, -0.2],
                "l_elbow": [-0.3, -0.3],
                "l_wrist": [-0.4, -0.4],
                "club_base": [0.35, -0.35],
                "club_com": [0.5, -0.6],
                "club_tip": [0.5, -1.0],
            }

    monkeypatch.setenv("PENDULUM_GOLFER_BACKEND", "rust")
    monkeypatch.setattr(native_backend, "_pendulum_core", StubCore)

    mapped = native_backend.golfer_forward_kinematics(np.zeros(8), golfer_params)

    assert mapped is not None
    assert mapped["origin"] == (0.0, 0.0)
    assert mapped["rs"] == (0.2, -0.2)
    assert mapped["grip_right"] == (0.4, -0.4)
    assert mapped["club_tip"] == (0.5, -1.0)


def test_native_double_wrappers_map_rust_outputs(
    monkeypatch: pytest.MonkeyPatch, double_params: PendulumParams
) -> None:
    class StubCore:
        @staticmethod
        def PyDoublePendulumParams(*args: float) -> tuple[float, ...]:
            return args

        @staticmethod
        def py_double_mass_matrix(
            q: list[float], params: tuple[float, ...]
        ) -> list[list[float]]:
            del q, params
            return [[5.0, 2.0], [2.0, 1.0]]

        @staticmethod
        def py_double_gravity_vector(
            q: list[float], params: tuple[float, ...]
        ) -> list[float]:
            del q, params
            return [3.0, 1.0]

        @staticmethod
        def py_double_coriolis(
            q: list[float], qdot: list[float], params: tuple[float, ...]
        ) -> list[float]:
            del q, qdot, params
            return [-21.0, 4.0]

        @staticmethod
        def py_double_forward_kinematics(
            q: list[float], params: tuple[float, ...]
        ) -> dict[str, float]:
            del q, params
            return {
                "wrist_x": 0.2,
                "wrist_y": -0.6,
                "club_tip_x": 0.5,
                "club_tip_y": -1.2,
                "theta1": 0.0,
                "theta2": 0.0,
            }

    monkeypatch.setenv("PENDULUM_DOUBLE_BACKEND", "rust")
    monkeypatch.setattr(native_backend, "_pendulum_core", StubCore)

    mass = native_backend.double_mass_matrix(0.0, double_params)
    gravity = native_backend.double_gravity_vector(0.0, 0.0, double_params)
    coriolis = native_backend.double_coriolis_vector(0.0, 0.0, 0.0, double_params)
    fk = native_backend.double_forward_kinematics(0.0, 0.0, double_params)

    assert mass is not None
    assert gravity is not None
    assert coriolis is not None
    assert fk is not None
    assert np.array_equal(mass, np.array([[5.0, 2.0], [2.0, 1.0]]))
    assert np.array_equal(gravity, np.array([3.0, 1.0]))
    assert np.array_equal(coriolis, np.array([-21.0, 4.0]))
    assert fk["tip"] == (0.5, -1.2)


def test_native_triple_wrappers_map_rust_outputs(
    monkeypatch: pytest.MonkeyPatch, triple_params: TriplePendulumParams
) -> None:
    class StubCore:
        @staticmethod
        def PyTriplePendulumParams(*args: float) -> tuple[float, ...]:
            return args

        @staticmethod
        def py_triple_mass_matrix(
            q: list[float], params: tuple[float, ...]
        ) -> list[list[float]]:
            del q, params
            return [[14.0, 8.0, 3.0], [8.0, 5.0, 2.0], [3.0, 2.0, 1.0]]

        @staticmethod
        def py_triple_gravity_vector(
            q: list[float], params: tuple[float, ...]
        ) -> list[float]:
            del q, params
            return [1.0, 2.0, 3.0]

        @staticmethod
        def py_triple_coriolis(
            q: list[float], qdot: list[float], params: tuple[float, ...]
        ) -> list[float]:
            del q, qdot, params
            return [-24.0, 3.0, 1.0]

        @staticmethod
        def py_triple_forward_kinematics(
            q: list[float], params: tuple[float, ...]
        ) -> dict[str, float]:
            del q, params
            return {
                "joint1_x": 0.1,
                "joint1_y": -0.6,
                "joint2_x": 0.2,
                "joint2_y": -1.2,
                "joint3_x": 0.3,
                "joint3_y": -1.8,
                "theta1": 0.0,
                "theta2": 0.0,
                "theta3": 0.0,
            }

    monkeypatch.setenv("PENDULUM_TRIPLE_BACKEND", "rust")
    monkeypatch.setattr(native_backend, "_pendulum_core", StubCore)

    mass = native_backend.triple_mass_matrix(0.0, 0.0, triple_params)
    gravity = native_backend.triple_gravity_vector(0.0, 0.0, 0.0, triple_params)
    coriolis = native_backend.triple_coriolis_vector(
        0.0, 0.0, 0.0, 0.0, 0.0, triple_params
    )
    fk = native_backend.triple_forward_kinematics(0.0, 0.0, 0.0, triple_params)

    assert mass is not None
    assert gravity is not None
    assert coriolis is not None
    assert fk is not None
    assert np.array_equal(
        mass,
        np.array([[14.0, 8.0, 3.0], [8.0, 5.0, 2.0], [3.0, 2.0, 1.0]]),
    )
    assert np.array_equal(gravity, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(coriolis, np.array([-24.0, 3.0, 1.0]))
    assert fk["wrist2"] == (0.2, -1.2)


def test_native_constrained_dynamics_respects_damping_guard(
    monkeypatch: pytest.MonkeyPatch, golfer_params: GolferParams
) -> None:
    monkeypatch.setenv("PENDULUM_GOLFER_BACKEND", "rust")
    monkeypatch.setattr(native_backend, "_pendulum_core", object())
    damped_params = GolferParams(**{**golfer_params.__dict__, "b_rs": 0.25})

    result = native_backend.golfer_constrained_dynamics(
        np.zeros(8),
        np.zeros(8),
        np.zeros(8),
        damped_params,
        alpha=5.0,
        beta=5.0,
    )

    assert result is None


def test_native_constrained_dynamics_maps_rust_outputs(
    monkeypatch: pytest.MonkeyPatch, golfer_params: GolferParams
) -> None:
    class StubCore:
        @staticmethod
        def PyGolferParams(*args: float) -> tuple[float, ...]:
            return args

        @staticmethod
        def py_golfer_constrained_dynamics(
            q: list[float],
            qdot: list[float],
            tau: list[float],
            params: tuple[float, ...],
            alpha: float,
            beta: float,
        ) -> tuple[list[float], list[float]]:
            del q, qdot, tau, params, alpha, beta
            return ([1.0] * 8, [2.0] * 4)

    monkeypatch.setenv("PENDULUM_GOLFER_BACKEND", "rust")
    monkeypatch.setattr(native_backend, "_pendulum_core", StubCore)

    result = native_backend.golfer_constrained_dynamics(
        np.zeros(8),
        np.zeros(8),
        np.zeros(8),
        golfer_params,
        alpha=5.0,
        beta=5.0,
    )

    assert result is not None
    qddot, lambda_forces = result
    assert np.array_equal(qddot, np.ones(8))
    assert np.array_equal(lambda_forces, np.full(4, 2.0))


def test_native_projection_wrappers_map_rust_outputs(
    monkeypatch: pytest.MonkeyPatch, golfer_params: GolferParams
) -> None:
    class StubCore:
        @staticmethod
        def PyGolferParams(*args: float) -> tuple[float, ...]:
            return args

        @staticmethod
        def py_golfer_project_to_constraints(
            q: list[float],
            params: tuple[float, ...],
            max_iters: int,
            tol: float,
        ) -> list[float]:
            del q, params, max_iters, tol
            return [0.25] * 8

        @staticmethod
        def py_golfer_project_velocity(
            q: list[float],
            qdot: list[float],
            params: tuple[float, ...],
        ) -> list[float]:
            del q, qdot, params
            return [0.5] * 8

    monkeypatch.setenv("PENDULUM_GOLFER_BACKEND", "rust")
    monkeypatch.setattr(native_backend, "_pendulum_core", StubCore)

    q_proj = native_backend.golfer_project_to_constraints(
        np.zeros(8), golfer_params, max_iters=5, tol=1e-6
    )
    qdot_proj = native_backend.golfer_project_velocity(
        np.zeros(8), np.zeros(8), golfer_params
    )

    assert q_proj is not None
    assert qdot_proj is not None
    assert np.array_equal(q_proj, np.full(8, 0.25))
    assert np.array_equal(qdot_proj, np.full(8, 0.5))


def test_simulate_double_works_with_valid_inputs() -> None:
    from double_pendulum_golf.physics import PendulumParams
    from double_pendulum_golf.native_backend import (
        simulate_double,
        double_native_enabled,
    )

    if not double_native_enabled():
        pytest.skip("Native backend disabled or unavailable")

    params = PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81)
    q0 = [0.0, 0.0]
    qdot0 = [0.0, 0.0]
    coeffs = [0.0, 0.0, 0.0, 0.0]
    n_coeffs = 2
    t_span = (0.0, 1.0)

    res = simulate_double(params, q0, qdot0, coeffs, n_coeffs, t_span, 100)
    assert res is not None
    t, states = res
    assert len(t) >= 2
    assert states.shape == (len(t), 4)
