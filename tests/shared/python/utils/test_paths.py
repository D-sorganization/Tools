"""Tests for upstream_drift_tools.utils.paths module.

Covers:
- get_repo_root with explicit start_path
- Marker-file detection (.git, pyproject.toml, tools.json)
- FileNotFoundError when no markers exist
- Search depth limit
"""

from __future__ import annotations

from pathlib import Path

import pytest
from upstream_drift_tools.utils.paths import get_repo_root


class TestGetRepoRoot:
    """Test get_repo_root function."""

    def test_finds_git_marker(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        result = get_repo_root(start_path=tmp_path)
        assert result == tmp_path

    def test_finds_pyproject_marker(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]")
        result = get_repo_root(start_path=tmp_path)
        assert result == tmp_path

    def test_finds_tools_json_marker(self, tmp_path: Path) -> None:
        (tmp_path / "tools.json").write_text("{}")
        result = get_repo_root(start_path=tmp_path)
        assert result == tmp_path

    def test_searches_upward(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        child = tmp_path / "src" / "module"
        child.mkdir(parents=True)
        result = get_repo_root(start_path=child)
        assert result == tmp_path

    def test_not_found_raises(self, tmp_path: Path) -> None:
        # Create a deeply nested path with no markers anywhere
        deep = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Repository root not found"):
            get_repo_root(start_path=deep)

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        result = get_repo_root(start_path=str(tmp_path))
        assert result == tmp_path

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        result = get_repo_root(start_path=tmp_path)
        assert result.is_absolute()
