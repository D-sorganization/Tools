"""Regression tests for the pendulum pytest discovery bridge."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from _pytest.config import Config

from conftest import PENDULUM_TESTS_DIR, pytest_ignore_collect
from tests.pendulum_simulator.conftest import BRIDGE_FILE_NAME, EMBEDDED_TESTS_DIR


def test_embedded_pendulum_suite_contains_expected_regression_target() -> None:
    """The bridge should point at the real embedded pendulum test suite."""
    discovered = {path.name for path in EMBEDDED_TESTS_DIR.glob("test_*.py")}

    assert "test_constants.py" in discovered
    assert len(discovered) >= 90


def test_bridge_placeholder_exists_under_top_level_tests() -> None:
    """Top-level pytest discovery should have a stable pendulum entrypoint."""
    bridge_file = (
        Path(__file__).resolve().parent / "pendulum_simulator" / BRIDGE_FILE_NAME
    )

    assert bridge_file.is_file()


def test_root_pytest_ignores_direct_pendulum_collection_by_default() -> None:
    """Default root-level pytest runs should not double-collect pendulum tests."""
    config = cast(
        Config,
        SimpleNamespace(rootpath=Path(__file__).resolve().parents[1], args=[]),
    )
    candidate = PENDULUM_TESTS_DIR / "test_constants.py"

    assert pytest_ignore_collect(candidate, config) is True


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
