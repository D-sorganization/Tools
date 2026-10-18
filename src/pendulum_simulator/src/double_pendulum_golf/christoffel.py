"""Christoffel symbols of the first kind for a configuration-dependent mass matrix.

Shared by the golfer Coriolis vector (``golfer_dynamics.analytical_coriolis``); the Rust
backend (``pendulum-core/src/golfer.rs``) and ``physics_golfer_jax.coriolis_jax`` implement
the same formula and finite-difference step (#5638, #5640).
"""

from __future__ import annotations

import numpy as np

__all__ = ["CORIOLIS_FD_STEP", "christoffel_first_kind"]

# Relative tolerance for the symmetry precondition of ``christoffel_first_kind``:
# finite differences of a symmetric matrix are symmetric to round-off only.
_SYMMETRY_RTOL = 1e-6

# Central-difference step for dM/dq in the Coriolis vector.  Optimal for double
# precision is ~eps^(1/3) ~ 6e-6; keep equal to CORIOLIS_FD_STEP in golfer.rs
# and physics_golfer_jax.py (pinned by tests).
CORIOLIS_FD_STEP = 1e-5


def christoffel_first_kind(dM: np.ndarray) -> np.ndarray:
    """Christoffel symbols of the first kind of the metric ``M(q)``.

    ``c[i, j, k] = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)`` with
    ``dM[i, j, k] = dM_ij/dq_k``.  This is the single definition of the formula; the
    Rust backend (``pendulum-core/src/golfer.rs``) implements the same contraction.

    Index map (``numpy.transpose`` semantics, ``out[x, y, z] = a[...]``):

    * ``dM.transpose(0, 2, 1)[i, j, k] = dM[i, k, j] = dM_ik/dq_j``
    * ``dM.transpose(2, 0, 1)[i, j, k] = dM[j, k, i] = dM_jk/dq_i``

    ``transpose(1, 2, 0)`` would give ``dM[k, i, j] = dM_ki/dq_j``, which equals the
    second term for a symmetric ``M`` and silently cancels it (#5640).

    Contract:
        Pre:  ``dM`` is a cubic ``(n, n, n)`` array that is symmetric in its first two
              indices (it is the derivative of a symmetric matrix).
        Post: ``c`` is symmetric in ``(j, k)`` and ``c[i,j,k] + c[j,i,k] == dM[i,j,k]``.
    """
    if not isinstance(dM, np.ndarray):
        raise TypeError("dM must be a numpy ndarray")
    if dM.ndim != 3 or not (dM.shape[0] == dM.shape[1] == dM.shape[2]):
        raise ValueError(f"dM must have shape (n, n, n), got {dM.shape}")
    scale = max(1.0, float(np.max(np.abs(dM))))
    if not np.allclose(dM, dM.transpose(1, 0, 2), rtol=0.0, atol=_SYMMETRY_RTOL * scale):
        raise ValueError("dM must be symmetric in its first two indices")
    result: np.ndarray = 0.5 * (dM + dM.transpose(0, 2, 1) - dM.transpose(2, 0, 1))
    return result
