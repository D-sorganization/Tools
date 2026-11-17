from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shared.python import trusted_git
from shared.python.codemap import api as codemap_api
from shared.python.codemap import indexer as codemap_indexer


def test_resolve_trusted_git_uses_absolute_env_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    git_path = tmp_path / "git"
    git_path.write_text("")
    git_path.chmod(0o755)
    monkeypatch.setattr(trusted_git, "_default_git_candidates", lambda env: ())
    monkeypatch.setenv(trusted_git.TRUSTED_GIT_ENV_VAR, str(git_path))

    assert trusted_git.resolve_trusted_git_executable() == str(git_path.resolve())


def test_resolve_trusted_git_rejects_relative_env_override(monkeypatch) -> None:
    monkeypatch.setattr(trusted_git, "_default_git_candidates", lambda env: ())
    monkeypatch.setenv(trusted_git.TRUSTED_GIT_ENV_VAR, "git")

    assert trusted_git.resolve_trusted_git_executable() is None


def test_discover_repo_root_falls_back_without_trusted_git(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "pkg" / "module"
    nested.mkdir(parents=True)
    (repo_root / ".git").mkdir()
    check_output = MagicMock(side_effect=AssertionError("git should not be called"))

    monkeypatch.setattr(codemap_api, "resolve_trusted_git_executable", lambda: None)
    monkeypatch.setattr(codemap_api.subprocess, "check_output", check_output)

    assert codemap_api.discover_repo_root(nested) == repo_root.resolve()


def test_git_changed_files_uses_trusted_absolute_git(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    check_output = MagicMock(return_value="a.py\nb.py\n")

    monkeypatch.setattr(
        codemap_indexer,
        "resolve_trusted_git_executable",
        lambda: "/trusted/bin/git",
    )
    monkeypatch.setattr(codemap_indexer.subprocess, "check_output", check_output)

    assert codemap_indexer._git_changed_files(repo_root, "HEAD~1") == ["a.py", "b.py"]
    check_output.assert_called_once_with(
        ["/trusted/bin/git", "diff", "--name-only", "HEAD~1..HEAD"],
        cwd=str(repo_root),
        stderr=subprocess.DEVNULL,
        text=True,
    )


def test_current_commit_uses_trusted_absolute_git(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    check_output = MagicMock(return_value="commit-hash-123\n")

    monkeypatch.setattr(
        codemap_indexer,
        "resolve_trusted_git_executable",
        lambda: "/trusted/bin/git",
    )
    monkeypatch.setattr(codemap_indexer.subprocess, "check_output", check_output)

    assert codemap_indexer._current_commit(repo_root) == "commit-hash-123"
    check_output.assert_called_once_with(
        ["/trusted/bin/git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        stderr=subprocess.DEVNULL,
        text=True,
    )


def test_current_commit_falls_back_without_trusted_git(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    check_output = MagicMock(side_effect=AssertionError("git should not be called"))

    monkeypatch.setattr(codemap_indexer, "resolve_trusted_git_executable", lambda: None)
    monkeypatch.setattr(codemap_indexer.subprocess, "check_output", check_output)


@pytest.mark.skipif(os.name == "nt", reason="POSIX execute bits and symlinks")
@pytest.mark.parametrize("candidate_kind", ["nonexecutable", "symlink_loop"])
def test_unusable_override_falls_back_to_executable_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate_kind: str
) -> None:
    """Broken override paths must not prevent a usable trusted fallback."""
    bad = tmp_path / "broken-git"
    if candidate_kind == "nonexecutable":
        bad.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        bad.chmod(0o644)
    else:
        other = tmp_path / "cycle"
        bad.symlink_to(other)
        other.symlink_to(bad)
    fallback = tmp_path / "working-git"
    fallback.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fallback.chmod(0o755)
    monkeypatch.setattr(trusted_git, "_default_git_candidates", lambda env: (fallback,))

    result = trusted_git.resolve_trusted_git_executable(
        {trusted_git.TRUSTED_GIT_ENV_VAR: str(bad)}
    )
    assert result == str(fallback.resolve())


def test_windows_discovery_preserves_regular_file_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows candidates retain their existing non-POSIX permission handling."""
    candidate = tmp_path / "git.exe"
    candidate.write_text("placeholder", encoding="utf-8")
    candidate.chmod(0o644)
    monkeypatch.setattr(trusted_git, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(trusted_git, "_default_git_candidates", lambda env: ())
    assert trusted_git.resolve_trusted_git_executable(
        {trusted_git.TRUSTED_GIT_ENV_VAR: str(candidate)}
    ) == str(candidate.resolve())
