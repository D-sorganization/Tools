"""Regression tests for #3736.

The redundant ``assert eomg is not None`` in ``_mr_kinematics.IKinBody`` was
removed because asserts are stripped under ``python -O`` and a following
``require(eomg > 0, ...)`` already covers the contract. These tests prove
that passing ``eomg=None`` still raises (the validation does not depend on
the deleted assert).
"""

from __future__ import annotations

import numpy as np
import pytest

from rotation_converter._mr_kinematics import IKinBody


def _simple_1r() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Blist = np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    M = np.eye(4)
    T_desired = np.eye(4)
    thetalist0 = np.array([0.0])
    return Blist, M, T_desired, thetalist0


def test_ikinbody_rejects_none_eomg_without_assert() -> None:
    """eomg=None must still raise even though the assert was removed."""
    Blist, M, T_desired, thetalist0 = _simple_1r()
    with pytest.raises((TypeError, ValueError)):
        IKinBody(Blist, M, T_desired, thetalist0, eomg=None)  # type: ignore[arg-type]


def test_ikinbody_rejects_nonpositive_eomg() -> None:
    """The positive-tolerance contract still holds."""
    Blist, M, T_desired, thetalist0 = _simple_1r()
    with pytest.raises((ValueError, AssertionError)):
        IKinBody(Blist, M, T_desired, thetalist0, eomg=-1.0)
