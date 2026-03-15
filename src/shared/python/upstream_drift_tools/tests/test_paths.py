"""Tests for upstream_drift_tools.utils.paths module.

Covers get_repo_root() with valid repos, missing repos, and default start_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from upstream_drift_tools.utils.paths import get_repo_root


class TestGetRepoRoot:
    def test_finds_repo_root_from_known_path(self):
        """Should find the repo root when starting from a directory inside it."""
        # We're running inside the Tools repo which has .git and pyproject.toml
        root = get_repo_root(Path(__file__).parent)
        assert root.exists()
        assert any(
            (root / marker).exists()
            for marker in (".git", "pyproject.toml", "tools.json")
        )

    def test_finds_from_nested_dir(self):
        """Works from deeply nested dir."""
        # Start from a deeply nested path inside the repo
        nested = Path(__file__).resolve().parent.parent.parent
        root = get_repo_root(nested)
        assert root.exists()

    def test_raises_when_no_repo_found(self, tmp_path):
        """Raises FileNotFoundError when no markers found up the tree."""
        # tmp_path is guaranteed to be outside the current repo
        orphan = tmp_path / "deeply" / "nested" / "dir"
        orphan.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Repository root not found"):
            get_repo_root(orphan)

    def test_default_path_finds_repo(self):
        """Calling with no arguments uses caller's file directory."""
        root = get_repo_root()
        assert root.exists()

    def test_string_path_accepted(self):
        """A string path is accepted (converted to Path internally)."""
        root = get_repo_root(str(Path(__file__).parent))
        assert root.exists()
