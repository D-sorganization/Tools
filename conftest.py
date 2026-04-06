"""Repository-level pytest hooks for cross-tree test discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
PENDULUM_TESTS_DIR = REPO_ROOT / "src" / "pendulum_simulator" / "tests"


def _path_is_within(candidate: Path, parent: Path) -> bool:
    """Return whether ``candidate`` is inside ``parent``."""
    try:
        candidate.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Avoid double-collecting the embedded pendulum suite from ``src/``.

    The suite is bridged into ``tests/pendulum_simulator`` so that
    ``pytest tests/`` includes it. If a developer explicitly targets the
    embedded directory, preserve the direct path-based behavior.
    """
    candidate = Path(collection_path)
    if not _path_is_within(candidate, PENDULUM_TESTS_DIR):
        return None

    explicit_targets = [
        (config.rootpath / arg).resolve()
        for arg in config.args
        if arg and not arg.startswith("-")
    ]
    if any(_path_is_within(target, PENDULUM_TESTS_DIR) for target in explicit_targets):
        return None
    return True
