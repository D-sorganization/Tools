"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import numpy as np
import pytest

from double_pendulum_golf.physics import PendulumParams

# ---------------------------------------------------------------------------
# Qt display availability detection
# ---------------------------------------------------------------------------


def _qt_available() -> bool:
    """Return True if a Qt platform backend can be initialised safely."""
    # The offscreen platform always works, even in headless CI.
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return True
    # Physical or virtual X11/Wayland display present.
    if bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY")):
        return True
    # Windows/macOS always have a native platform.
    return sys.platform in ("win32", "darwin")


@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication instance for Qt widget tests.

    Skips automatically when no Qt platform backend is available
    (e.g. a headless runner without QT_QPA_PLATFORM=offscreen).
    """
    if not _qt_available():
        pytest.skip("No Qt platform available (set QT_QPA_PLATFORM=offscreen)")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def default_params() -> PendulumParams:
    """Standard test parameters: golf-like arm + club."""
    return PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0)


@pytest.fixture
def equal_params() -> PendulumParams:
    """Equal-mass, equal-length pendulum for symmetry tests."""
    return PendulumParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0)


@pytest.fixture
def aligned_state() -> np.ndarray:
    """Both segments straight down, positive velocities."""
    return np.array([0.0, 0.0, 2.0, 1.0])


@pytest.fixture
def cocked_state() -> np.ndarray:
    """Arm back, club cocked (golf backswing position)."""
    return np.array([np.radians(120), np.radians(-90), 0.0, 0.0])


@pytest.fixture
def zero_torque() -> Callable[[float], tuple[float, float]]:
    """No applied torques."""
    return lambda t: (0.0, 0.0)


@pytest.fixture
def constant_shoulder_torque() -> Callable[[float], tuple[float, float]]:
    """Constant shoulder torque, zero wrist."""
    return lambda t: (-25.0, 0.0)
