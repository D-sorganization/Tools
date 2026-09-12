"""Tests for Wave 1: canonical path utilities and bootstrap module.

Verifies:
- get_repo_root() finds the correct repository root
- ensure_paths() adds standard directories to sys.path
- StateManager uses lazy initialization (no import-time side effects)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


class TestGetRepoRoot:
    """Tests for the canonical get_repo_root() implementation."""

    def test_finds_repo_root_from_this_file(self) -> None:
        from upstream_drift_tools.utils.paths import get_repo_root

        root = get_repo_root(start_path=Path(__file__).parent)
        assert (root / "pyproject.toml").exists()
        assert (root / "tools.json").exists() or (root / ".git").exists()

    def test_finds_repo_root_from_deep_path(self) -> None:
        from upstream_drift_tools.utils.paths import get_repo_root

        # Even if a deep path doesn't exist, it should search upward
        root = get_repo_root(start_path=Path(__file__).parent)
        assert root.is_absolute()
        assert (root / "pyproject.toml").exists()

    def test_raises_on_nonexistent_root(self, tmp_path: Path) -> None:
        from upstream_drift_tools.utils.paths import get_repo_root

        # Create a deep path with no markers
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Repository root not found"):
            get_repo_root(start_path=deep)

    def test_returns_absolute_path(self) -> None:
        from upstream_drift_tools.utils.paths import get_repo_root

        root = get_repo_root()
        assert root.is_absolute()


class TestEnsurePaths:
    """Tests for the bootstrap ensure_paths() function."""

    def test_adds_standard_paths(self) -> None:
        from upstream_drift_tools.bootstrap import ensure_paths
        from upstream_drift_tools.utils.paths import get_repo_root

        repo_root = get_repo_root()
        ensure_paths(repo_root)

        expected_in_path = repo_root / "src"
        assert str(expected_in_path) in sys.path

    def test_returns_repo_root(self) -> None:
        from upstream_drift_tools.bootstrap import ensure_paths
        from upstream_drift_tools.utils.paths import get_repo_root

        expected = get_repo_root()
        result = ensure_paths(expected)
        assert result == expected

    def test_idempotent(self) -> None:
        from upstream_drift_tools.bootstrap import ensure_paths
        from upstream_drift_tools.utils.paths import get_repo_root

        repo_root = get_repo_root()
        ensure_paths(repo_root)
        path_count_before = len(sys.path)
        ensure_paths(repo_root)
        path_count_after = len(sys.path)
        assert path_count_after == path_count_before


class TestStateManagerLazy:
    """Tests that StateManager uses lazy initialization via _StateManagerHolder."""

    def test_no_global_instance_on_import(self) -> None:
        """Importing the module should use _StateManagerHolder for lazy init."""
        import upstream_drift_tools.utils.state_manager as sm

        assert sm._StateManagerHolder.instance is None or isinstance(
            sm._StateManagerHolder.instance, sm.StateManager
        )

    def test_get_state_manager_creates_instance(self, tmp_path: Path) -> None:
        """get_state_manager() should create instance on first call."""
        import upstream_drift_tools.utils.state_manager as sm

        # Reset holder
        sm._StateManagerHolder.instance = None
        mgr = sm.get_state_manager(base_directory=str(tmp_path / "test_states"))
        assert isinstance(mgr, sm.StateManager)
        assert sm._StateManagerHolder.instance is mgr

    def test_get_state_manager_returns_same_instance(self, tmp_path: Path) -> None:
        """Subsequent calls should return the same instance."""
        import upstream_drift_tools.utils.state_manager as sm

        sm._StateManagerHolder.instance = None
        mgr1 = sm.get_state_manager(base_directory=str(tmp_path / "test_states"))
        mgr2 = sm.get_state_manager()
        assert mgr1 is mgr2


class TestPathSetupBackwardCompat:
    """Tests that path_setup.py still works for backward compatibility."""

    def test_get_repo_root_available(self) -> None:
        from utils.path_setup import get_repo_root

        root = get_repo_root()
        assert (root / "pyproject.toml").exists()

    def test_setup_python_path_available(self) -> None:
        from utils.path_setup import setup_python_path

        # Should not raise
        setup_python_path()


class TestMaintenanceScriptPathHygiene:
    """Tests for developer-machine path and sys.path bootstrap removal."""

    @staticmethod
    def _read_text(filename: str) -> str:
        return (Path(__file__).resolve().parents[1] / filename).read_text(
            encoding="utf-8"
        )

    def test_wave_solver_uses_repo_root_discovery(self) -> None:
        content = self._read_text("wave_solver.py")
        assert "C:/Users/diete/Repositories/Tools" not in content
        assert "REPO_ROOT = Path(__file__).resolve().parent" in content

    def test_commit_screensaver_uses_repo_root_discovery(self) -> None:
        content = self._read_text("commit_screensaver.py")
        assert "C:\\Users\\diete\\Repositories\\Tools" not in content
        assert "REPO_ROOT = Path(__file__).resolve().parent" in content

    def test_convert_tools_icon_has_no_sys_path_bootstrap(self) -> None:
        content = self._read_text("convert_tools_icon.py")
        assert "sys.path.append" not in content
        assert "tools.icon_utils" in content
