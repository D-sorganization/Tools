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
