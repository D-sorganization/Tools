"""Optional Rust-backed kernels for pendulum simulation.

The existing ``pendulum_core`` extension already exposes several golfer-model
physics kernels. This module provides a narrow adapter around those bindings so
the Python desktop app can opt into the native implementation without changing
its public API.

The backend is intentionally opt-in. The golfer model is the primary
performance hotspot identified in the deep review, while the double and triple
models still need explicit parity validation before they should be promoted to
native execution by default.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .physics_golfer import GolferParams

_GOLFER_BACKEND_ENV = "PENDULUM_GOLFER_BACKEND"
_WARNED_CALLS: set[str] = set()

try:
    import pendulum_core as _pendulum_core

    _NATIVE_IMPORT_ERROR: str | None = None
except ImportError as exc:
    _pendulum_core = None
    _NATIVE_IMPORT_ERROR = str(exc)


def _warn_once(call_name: str, exc: Exception) -> None:
    """Warn once per native call site, then fall back to Python."""
    if call_name in _WARNED_CALLS:
        return
    _WARNED_CALLS.add(call_name)
    warnings.warn(
        (
            f"Rust backend call '{call_name}' failed with "
            f"{exc!r}; falling back to the Python implementation."
        ),
        RuntimeWarning,
        stacklevel=2,
    )


def _truncate_q(q: np.ndarray) -> np.ndarray:
    """Normalize q to the 8 generalized coordinates expected by the Rust core."""
    q_arr = np.asarray(q, dtype=float)
    if q_arr.shape[0] > 8:
        q_arr = q_arr[:8]
    assert q_arr.shape == (8,), f"q must have shape (8,), got {q_arr.shape}"
    return q_arr


def golfer_backend_mode() -> str:
    """Return the configured golfer backend mode."""
    mode = os.getenv(_GOLFER_BACKEND_ENV, "python").strip().lower()
    return mode if mode in {"python", "rust"} else "python"


def golfer_native_available() -> bool:
    """Whether the compiled ``pendulum_core`` Python extension is importable."""
    return _pendulum_core is not None


def golfer_native_enabled() -> bool:
    """Whether golfer kernels should use the Rust extension."""
    return golfer_backend_mode() == "rust" and golfer_native_available()


def get_native_backend_info() -> dict[str, object]:
    """Return backend configuration and availability details."""
    return {
        "configured_backend": golfer_backend_mode(),
        "native_available": golfer_native_available(),
        "native_import_error": _NATIVE_IMPORT_ERROR,
        "supported_models": {"golfer": True, "double": False, "triple": False},
    }


def _to_rust_golfer_params(params: GolferParams) -> Any:
    """Convert the Python golfer params dataclass to the PyO3 wrapper type."""
    if _pendulum_core is None:
        raise RuntimeError("pendulum_core is not available")

    return _pendulum_core.PyGolferParams(
        params.L_hub,
        params.m_hub,
        params.d_rs,
        params.d_ls,
        params.L_r_upper,
        params.m_r_upper,
        params.L_r_fore,
        params.m_r_fore,
        params.L_l_upper,
        params.m_l_upper,
        params.L_l_fore,
        params.m_l_fore,
        params.L_club,
        params.m_club,
        params.m_clubhead,
        params.grip_right,
        params.grip_left,
        params.g,
    )


def golfer_mass_matrix(q: np.ndarray, params: GolferParams) -> np.ndarray | None:
    """Return the native golfer mass matrix, or ``None`` if disabled/unavailable."""
    if not golfer_native_enabled():
        return None

    try:
        q_arr = _truncate_q(q)
        result = _pendulum_core.py_golfer_mass_matrix(
            q_arr.tolist(), _to_rust_golfer_params(params)
        )
        return np.array(result, dtype=float)
    except Exception as exc:  # pragma: no cover - exercised when extension exists
        _warn_once("golfer_mass_matrix", exc)
        return None


def golfer_gravity_vector(q: np.ndarray, params: GolferParams) -> np.ndarray | None:
    """Return the native golfer gravity vector, or ``None`` if disabled/unavailable."""
    if not golfer_native_enabled():
        return None

    try:
        q_arr = _truncate_q(q)
        result = _pendulum_core.py_golfer_gravity_vector(
            q_arr.tolist(), _to_rust_golfer_params(params)
        )
        return np.array(result, dtype=float)
    except Exception as exc:  # pragma: no cover - exercised when extension exists
        _warn_once("golfer_gravity_vector", exc)
        return None


def golfer_forward_kinematics(
    q: np.ndarray, params: GolferParams
) -> dict[str, tuple[float, float]] | None:
    """Return native golfer forward kinematics mapped to Python GUI keys."""
    if not golfer_native_enabled():
        return None

    try:
        q_arr = _truncate_q(q)
        result = _pendulum_core.py_golfer_forward_kinematics(
            q_arr.tolist(), _to_rust_golfer_params(params)
        )
    except Exception as exc:  # pragma: no cover - exercised when extension exists
        _warn_once("golfer_forward_kinematics", exc)
        return None

    theta_club = float(q_arr[7])
    club_dx = float(np.sin(theta_club))
    club_dy = float(-np.cos(theta_club))
    club_base = tuple(result["club_base"])
    grip_left = (
        float(club_base[0] + params.grip_left * club_dx),
        float(club_base[1] - params.grip_left * club_dy),
    )

    return {
        "origin": (0.0, 0.0),
        "hub": tuple(result["hub"]),
        "rs": tuple(result["r_shoulder"]),
        "re": tuple(result["r_elbow"]),
        "rh": tuple(result["r_wrist"]),
        "ls": tuple(result["l_shoulder"]),
        "le": tuple(result["l_elbow"]),
        "lh": tuple(result["l_wrist"]),
        "club_base": tuple(result["club_base"]),
        "club_tip": tuple(result["club_tip"]),
        "grip_right": tuple(result["r_wrist"]),
        "grip_left": grip_left,
    }
