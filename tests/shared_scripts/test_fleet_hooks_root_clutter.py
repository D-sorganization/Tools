"""Coverage for ``fleet_hooks.check_root_clutter``.

``shared_scripts/fleet_hooks.py`` is a *fleet-wide* pre-commit hook: every repo
that consumes the Repository_Management templates runs it. It had no tests at
all, which is how a change that would have inverted its default (rejecting
anything not explicitly allowlisted, rather than rejecting things that look like
scratch output) could be written without anyone noticing — see issue #4486.

These tests pin the two properties that matter:

* the allowlist covers the root entries a Rust-bearing repo legitimately has, and
* the check stays *deny-scratch*, not *deny-unless-allowlisted*.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / "shared_scripts" / "fleet_hooks.py"


def _load_fleet_hooks() -> ModuleType:
    """Import ``shared_scripts/fleet_hooks.py`` by path.

    ``shared_scripts`` is not a package (no ``__init__.py``) and is synced from
    Repository_Management, so it is loaded by file location rather than by
    adding packaging files this repo does not own.
    """
    spec = importlib.util.spec_from_file_location("_fleet_hooks_under_test", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fleet_hooks = _load_fleet_hooks()


def _run(
    monkeypatch: pytest.MonkeyPatch, staged: list[str], *, warn_only: bool = False
) -> int:
    """Run ``check_root_clutter`` over a fake staged-file list."""
    monkeypatch.setattr(fleet_hooks, "staged_files", lambda: staged)
    args = argparse.Namespace(warn_only=warn_only)
    return fleet_hooks.check_root_clutter(args)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["pyproject.toml", "README.md", "uv.lock", "Cargo.toml", "Cargo.lock", "target"],
)
def test_allowlisted_root_entry_passes(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    assert _run(monkeypatch, [name]) == 0


@pytest.mark.unit
@pytest.mark.parametrize("name", ["Cargo.toml", "Cargo.lock", "target"])
def test_rust_root_entries_are_allowlisted(name: str) -> None:
    """The Rust entries from issue #4486 are present in the allowlist.

    This is asserted against the set rather than through ``check_root_clutter``
    on purpose. Under the current deny-scratch default these three names pass
    the check whether or not they are allowlisted — none of them carries a
    scratch suffix — so a behavioural test would pass without the change and
    prove nothing. They are staged here so the allowlist is already right if
    the stricter default is ever adopted.
    """
    assert name in fleet_hooks.ROOT_ALLOWLIST


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["debug.log", "scratch.tmp", "notes.bak", "dump.zip", "dump.7z", "DEBUG.LOG"],
)
def test_root_scratch_output_fails(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    assert _run(monkeypatch, [name]) == 1


@pytest.mark.unit
def test_nested_paths_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check only polices the repo root, never subdirectories."""
    staged = [
        "logs/debug.log",
        "src/tool/scratch.tmp",
        "target/release/build.log",
        "vendor/archive.zip",
    ]
    assert _run(monkeypatch, staged) == 0


@pytest.mark.unit
def test_nested_paths_are_ignored_with_windows_separators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run(monkeypatch, [r"logs\debug.log"]) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [".gitattributes", ".gitmodules", ".dockerignore", ".editorconfig", "noxfile.py"],
)
def test_unlisted_but_ordinary_root_file_passes(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The check rejects scratch output — not everything off the allowlist.

    Regression guard for issue #4486: a proposed catch-all would have failed
    30-40 tracked, entirely ordinary root files in every fleet repo. If the
    stricter default is ever wanted it has to land deliberately, behind
    ``--warn-only`` first, with the allowlist expanded to match.
    """
    assert _run(monkeypatch, [name]) == 0


@pytest.mark.unit
def test_warn_only_reports_but_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch, ["debug.log"], warn_only=True) == 0


@pytest.mark.unit
def test_mixed_staging_fails_on_the_scratch_file_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    staged = ["Cargo.toml", "src/lib.rs", "debug.log"]
    with caplog.at_level("ERROR", logger=fleet_hooks.logger.name):
        assert _run(monkeypatch, staged) == 1
    messages = caplog.text
    assert "debug.log" in messages
    assert "Cargo.toml" not in messages


@pytest.mark.unit
def test_clean_staging_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch, []) == 0
