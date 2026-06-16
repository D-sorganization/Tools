"""Unit tests for sidekick.bootstrap (issues #3032).

Verifies the ensure_paths function's path injection logic without relying
on any Qt or heavy dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SHARED = Path(__file__).resolve().parents[4] / "src" / "shared" / "python"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))


# ---------------------------------------------------------------------------
# Module-level import fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bootstrap():  # type: ignore[no-untyped-def]
    """Import and return the bootstrap module."""
    import importlib

    return importlib.import_module("sidekick.bootstrap")


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_bootstrap_all_exports(bootstrap) -> None:  # type: ignore[no-untyped-def]
    """bootstrap.__all__ exports exactly ensure_paths."""
    assert bootstrap.__all__ == ["ensure_paths"]


def test_ensure_paths_is_callable(bootstrap) -> None:  # type: ignore[no-untyped-def]
    """ensure_paths is a callable."""
    assert callable(bootstrap.ensure_paths)


# ---------------------------------------------------------------------------
# ensure_paths — functional tests with a real temporary tree
# ---------------------------------------------------------------------------


def test_ensure_paths_with_explicit_root(bootstrap, tmp_path: Path) -> None:
    """ensure_paths with an explicit repo_root returns a resolved Path."""
    # Create the expected subdirectory structure
    src = tmp_path / "src"
    src.mkdir(parents=True)

    result = bootstrap.ensure_paths(repo_root=tmp_path)

    assert isinstance(result, Path)
    assert result == tmp_path.resolve()


def test_ensure_paths_adds_src(bootstrap, tmp_path: Path) -> None:
    """ensure_paths inserts src into sys.path when it exists."""
    target = tmp_path / "src"
    target.mkdir(parents=True)

    original_path = sys.path.copy()
    try:
        bootstrap.ensure_paths(repo_root=tmp_path)
        assert str(target) in sys.path
    finally:
        # Restore sys.path to avoid test pollution
        sys.path[:] = original_path


def test_ensure_paths_adds_python_src(bootstrap, tmp_path: Path) -> None:
    """ensure_paths inserts src/python/src into sys.path when it exists."""
    src = tmp_path / "src" / "python" / "src"
    src.mkdir(parents=True)

    original_path = sys.path.copy()
    try:
        bootstrap.ensure_paths(repo_root=tmp_path)
        assert str(src) in sys.path
    finally:
        sys.path[:] = original_path


def test_ensure_paths_idempotent(bootstrap, tmp_path: Path) -> None:
    """ensure_paths does not add the same path twice."""
    target = tmp_path / "src"
    target.mkdir(parents=True)

    original_path = sys.path.copy()
    try:
        bootstrap.ensure_paths(repo_root=tmp_path)
        count_before = sys.path.count(str(target))
        bootstrap.ensure_paths(repo_root=tmp_path)
        count_after = sys.path.count(str(target))
        assert count_before == count_after
    finally:
        sys.path[:] = original_path


def test_ensure_paths_skips_nonexistent_dirs(bootstrap, tmp_path: Path) -> None:
    """ensure_paths does not add non-existent directories to sys.path."""
    # tmp_path exists but has no src/ subdirectory
    absent = str(tmp_path / "src" / "shared" / "python")

    original_path = sys.path.copy()
    try:
        bootstrap.ensure_paths(repo_root=tmp_path)
        assert absent not in sys.path
    finally:
        sys.path[:] = original_path


def test_ensure_paths_accepts_str_root(bootstrap, tmp_path: Path) -> None:
    """ensure_paths accepts a string repo_root and returns a Path."""
    result = bootstrap.ensure_paths(repo_root=str(tmp_path))
    assert isinstance(result, Path)
