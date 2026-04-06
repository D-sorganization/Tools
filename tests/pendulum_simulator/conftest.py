"""Bridge embedded pendulum tests into the top-level test tree."""

from __future__ import annotations

from pathlib import Path

import pytest

EMBEDDED_TESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "pendulum_simulator" / "tests"
)
BRIDGE_FILE_NAME = "test_embedded_suite.py"


class EmbeddedPendulumSuite(pytest.File):
    """Collect the legacy embedded pendulum tests through ``tests/``."""

    def collect(self) -> list[pytest.Module]:
        return [
            pytest.Module.from_parent(self, path=test_file)
            for test_file in sorted(EMBEDDED_TESTS_DIR.glob("test_*.py"))
        ]


def pytest_collect_file(
    file_path: Path, parent: pytest.Collector
) -> EmbeddedPendulumSuite | None:
    """Route the placeholder bridge file to the embedded suite collector."""
    if file_path.name != BRIDGE_FILE_NAME:
        return None
    return EmbeddedPendulumSuite.from_parent(parent, path=file_path)
