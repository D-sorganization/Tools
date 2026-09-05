"""Tests for upstream_drift_tools.bootstrap module.

Covers:
- ensure_paths with explicit repo_root
- Adds standard directories to sys.path
- Idempotent (doesn't duplicate paths)
- Returns resolved repo root
"""

from __future__ import annotations

import sys
from pathlib import Path

from upstream_drift_tools.bootstrap import ensure_paths


class TestEnsurePaths:
    """Test the ensure_paths bootstrap function."""

    def test_returns_resolved_root(self, tmp_path: Path) -> None:
        result = ensure_paths(repo_root=tmp_path)
        assert result == tmp_path.resolve()

    def test_adds_standard_paths(self, tmp_path: Path) -> None:
        # Create the standard directories so they are eligible
        shared = tmp_path / "src" / "shared" / "python"
        shared.mkdir(parents=True)
        src = tmp_path / "src"
        py_src = tmp_path / "src" / "python" / "src"
        py_src.mkdir(parents=True)

        # Record original sys.path length
        original = list(sys.path)
        try:
            ensure_paths(repo_root=tmp_path)
            assert str(shared) not in sys.path
            assert str(src) in sys.path
            assert str(py_src) in sys.path
        finally:
            # Restore sys.path
            sys.path[:] = original

    def test_idempotent(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir(parents=True)

        original = list(sys.path)
        try:
            ensure_paths(repo_root=tmp_path)
            count_1 = sys.path.count(str(src))
            ensure_paths(repo_root=tmp_path)
            count_2 = sys.path.count(str(src))
            assert count_1 == count_2 == 1
        finally:
            sys.path[:] = original

    def test_skips_nonexistent_dirs(self, tmp_path: Path) -> None:
        # Don't create any standard dirs
        original = list(sys.path)
        try:
            ensure_paths(repo_root=tmp_path)
            # None of the standard paths should have been added
            shared = str(tmp_path / "src" / "shared" / "python")
            assert shared not in sys.path
        finally:
            sys.path[:] = original

    def test_accepts_string_root(self, tmp_path: Path) -> None:
        result = ensure_paths(repo_root=str(tmp_path))
        assert isinstance(result, Path)
        assert result == tmp_path.resolve()
