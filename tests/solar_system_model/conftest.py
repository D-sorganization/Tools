"""Bridge embedded solar-system tests into the top-level test tree."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

EMBEDDED_TESTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "solar_system_model"
    / "solar_system"
    / "tests"
)
BRIDGE_DIR = Path(__file__).resolve().parent


class EmbeddedSolarSystemSuite(pytest.Directory):
    """Collect the legacy embedded solar-system tests through ``tests/``."""

    def collect(self) -> list[pytest.Module]:
        return [
            pytest.Module.from_parent(self, path=test_file)
            for test_file in sorted(EMBEDDED_TESTS_DIR.glob("test_*.py"))
        ]


def pytest_collect_directory(
    path: Path, parent: pytest.Collector
) -> EmbeddedSolarSystemSuite | None:
    """Route the bridge directory to the embedded suite collector."""
    if path.resolve() != BRIDGE_DIR.resolve():
        return None
    return cast(
        EmbeddedSolarSystemSuite,
        EmbeddedSolarSystemSuite.from_parent(parent, path=path),
    )
