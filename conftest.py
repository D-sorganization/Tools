"""Repository-level pytest hooks for cross-tree test discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
BRIDGED_EMBEDDED_TEST_DIRS = {
    REPO_ROOT / "src" / "pendulum_simulator" / "tests",
    REPO_ROOT / "src" / "solar_system_model" / "solar_system" / "tests",
}


def _path_is_within(candidate: Path, parent: Path) -> bool:
    """Return whether ``candidate`` is inside ``parent``."""
    try:
        candidate.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Avoid double-collecting embedded suites that are bridged into ``tests/``.

    If a developer explicitly targets one of the embedded directories, preserve
    the direct path-based behavior.
    """
    candidate = Path(collection_path)
    explicit_targets = [
        (config.rootpath / arg).resolve()
        for arg in config.args
        if arg and not arg.startswith("-")
    ]

    for embedded_tests_dir in BRIDGED_EMBEDDED_TEST_DIRS:
        if not _path_is_within(candidate, embedded_tests_dir):
            continue
        if any(
            _path_is_within(target, embedded_tests_dir) for target in explicit_targets
        ):
            return None
        return True

    return None
