"""Prettier must not rewrite files owned by deterministic generators (#6171)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _prettier_hook() -> dict[str, Any]:
    config = cast(
        dict[str, Any],
        yaml.safe_load(
            (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        ),
    )
    for repo in config["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "prettier":
                return cast(dict[str, Any], hook)
    raise AssertionError("prettier hook is missing")


def _prettier_exclude() -> re.Pattern[str]:
    return re.compile(cast(str, _prettier_hook()["exclude"]))


def test_prettier_runs_only_at_commit_time() -> None:
    """At pre-push, prettier would rewrite other commits' files in the range."""
    assert _prettier_hook().get("stages") == ["pre-commit"]


@pytest.mark.parametrize(
    "path",
    [
        # collate-changes.yml writes it; the baselines contract compares it
        # to SPEC.md rows, so prettier's underscore escaping breaks that test.
        "docs/development/DEVELOPMENT_LOG.md",
        "manuals/tools/manifests/module-inventory.json",
        "manuals/tools/manifests/module-inventory/entries-scripts.json",
    ],
)
def test_generator_owned_files_are_excluded_from_prettier(path: str) -> None:
    assert _prettier_exclude().search(path), f"prettier would rewrite {path}"


@pytest.mark.parametrize(
    "path", ["README.md", "changes/README.md", "docs/development/OTHER.md"]
)
def test_hand_written_markdown_still_formatted(path: str) -> None:
    assert not _prettier_exclude().search(path), f"{path} must stay formatted"
