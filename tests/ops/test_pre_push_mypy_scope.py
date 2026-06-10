"""Pre-push hook contracts for changed-file mypy checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _pre_commit_config() -> dict[str, Any]:
    return yaml.safe_load(
        (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )


def _mypy_hook() -> dict[str, Any]:
    for repo in _pre_commit_config()["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "mypy":
                return hook
    raise AssertionError("mypy pre-push hook is missing")


def test_pre_push_mypy_is_changed_file_delta_scoped() -> None:
    """Pre-push mypy must not fail clean pushes on unrelated imported debt."""
    hook = _mypy_hook()

    assert hook["stages"] == ["pre-push"]
    assert hook["files"] == "^src/"
    assert hook.get("pass_filenames", True) is True
    assert "--follow-imports=skip" in hook["args"]
