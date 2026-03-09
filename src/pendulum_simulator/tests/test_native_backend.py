"""Tests for the optional Rust-backed golfer kernel adapter."""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf import native_backend
from double_pendulum_golf.physics_golfer import (
    GolferParams,
    forward_kinematics,
    get_native_backend_info,
    gravity_vector,
    mass_matrix,
)


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
    monkeypatch.delenv("PENDULUM_GOLFER_BACKEND", raising=False)
    info = get_native_backend_info()

    assert info["configured_backend"] == "python"
    assert info["supported_models"] == {
        "golfer": True,
        "double": False,
        "triple": False,
    }
    assert info["supports_constraint_dynamics"] is True


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
