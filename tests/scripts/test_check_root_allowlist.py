"""Tests for scripts/check_root_allowlist.py (Tools#4917 root hygiene gate)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.check_root_allowlist import (
    ROOT_ALLOWLIST,
    disallowed_entries,
    main,
    tracked_top_level_entries,
)

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable required"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _track(repo: Path, *relative_paths: str) -> None:
    for relative in relative_paths:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{relative}\n", encoding="utf-8")
    _git(repo, "add", "--", *relative_paths)
    _git(repo, "commit", "-q", "-m", "track")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _track(repo, "README.md", "src/pkg/module.py", "tests/test_module.py")
    return repo


def test_disallowed_entries_returns_sorted_offenders_only() -> None:
    assert disallowed_entries(("src", "zeta.md", "alpha.txt", "tests")) == (
        "alpha.txt",
        "zeta.md",
    )
    assert disallowed_entries(("src", "tests")) == ()


def test_tracked_top_level_entries_collapses_nested_paths(git_repo: Path) -> None:
    assert tracked_top_level_entries(git_repo) == ("README.md", "src", "tests")


def test_passes_when_tracked_entries_are_a_subset_of_the_allowlist(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(git_repo)]) == 0

    assert "passed (3 tracked top-level entries)" in capsys.readouterr().out


def test_fails_and_names_offender_when_extra_top_level_file_is_tracked(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _track(git_repo, "scratch_notes.md", "src/pkg/other.py")

    assert main(["--root", str(git_repo)]) == 1

    captured = capsys.readouterr()
    assert "- scratch_notes.md" in captured.err
    assert "src" not in captured.err.replace("scripts/check_root_allowlist", "")


def test_list_prints_entries_without_gating(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _track(git_repo, "scratch_notes.md")

    assert main(["--root", str(git_repo), "--list"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "README.md",
        "scratch_notes.md",
        "src",
        "tests",
    ]


def test_allowlist_matches_this_repository() -> None:
    assert disallowed_entries(tracked_top_level_entries(ROOT)) == ()
    assert isinstance(ROOT_ALLOWLIST, frozenset)


def test_gate_is_wired_into_pre_commit_and_topology_check() -> None:
    pre_commit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    topology = (ROOT / "scripts" / "check_repo_topology.py").read_text(encoding="utf-8")

    assert "python scripts/check_root_allowlist.py" in pre_commit
    assert "check_root_allowlist" in topology
