"""Regression tests for embedded pytest discovery bridges."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from _pytest.config import Config

from conftest import BRIDGED_EMBEDDED_TEST_DIRS, pytest_ignore_collect
from tests.pendulum_simulator.conftest import BRIDGE_DIR as PENDULUM_BRIDGE_DIR
from tests.pendulum_simulator.conftest import EMBEDDED_TESTS_DIR as PENDULUM_TESTS_DIR
from tests.solar_system_model.conftest import BRIDGE_DIR as SOLAR_BRIDGE_DIR
from tests.solar_system_model.conftest import (
    EMBEDDED_TESTS_DIR as SOLAR_SYSTEM_TESTS_DIR,
)


def test_embedded_pendulum_suite_contains_expected_regression_target() -> None:
    """The bridge should point at the real embedded pendulum test suite."""
    discovered = {path.name for path in PENDULUM_TESTS_DIR.glob("test_*.py")}

    assert "test_constants.py" in discovered
    assert len(discovered) >= 90


def test_bridge_directories_exist_under_top_level_tests() -> None:
    """Top-level pytest discovery should have stable embedded-suite entrypoints."""
    bridge_directories = [PENDULUM_BRIDGE_DIR, SOLAR_BRIDGE_DIR]

    assert all(path.is_dir() for path in bridge_directories)


def test_embedded_solar_suite_contains_expected_regression_target() -> None:
    """The bridge should point at the real embedded solar-system test suite."""
    discovered = {path.name for path in SOLAR_SYSTEM_TESTS_DIR.glob("test_*.py")}

    assert "test_orbital_mechanics.py" in discovered
    assert len(discovered) >= 6


def test_root_pytest_ignores_direct_embedded_collection_by_default() -> None:
    """Default root-level pytest runs should not double-collect bridged tests."""
    config = cast(
        Config,
        SimpleNamespace(rootpath=Path(__file__).resolve().parents[1], args=[]),
    )
    candidates = [
        PENDULUM_TESTS_DIR / "test_constants.py",
        SOLAR_SYSTEM_TESTS_DIR / "test_orbital_mechanics.py",
    ]

    assert all(
        pytest_ignore_collect(candidate, config) is True for candidate in candidates
    )


def test_root_pytest_keeps_explicit_pendulum_paths() -> None:
    """Explicit pendulum test invocations should still work unchanged."""
    repo_root = Path(__file__).resolve().parents[1]
    config = cast(
        Config,
        SimpleNamespace(
            rootpath=repo_root,
            args=["src/pendulum_simulator/tests/test_constants.py"],
        ),
    )
    candidate = PENDULUM_TESTS_DIR / "test_constants.py"

    assert pytest_ignore_collect(candidate, config) is None


def test_root_pytest_keeps_explicit_solar_paths() -> None:
    """Explicit solar-system test invocations should still work unchanged."""
    repo_root = Path(__file__).resolve().parents[1]
    config = cast(
        Config,
        SimpleNamespace(
            rootpath=repo_root,
            args=[
                "src/solar_system_model/solar_system/tests/test_orbital_mechanics.py"
            ],
        ),
    )
    candidate = SOLAR_SYSTEM_TESTS_DIR / "test_orbital_mechanics.py"

    assert pytest_ignore_collect(candidate, config) is None


def test_root_bridge_registry_tracks_supported_embedded_suites() -> None:
    """The root hook should explicitly enumerate every bridged embedded suite."""
    assert BRIDGED_EMBEDDED_TEST_DIRS == {PENDULUM_TESTS_DIR, SOLAR_SYSTEM_TESTS_DIR}
