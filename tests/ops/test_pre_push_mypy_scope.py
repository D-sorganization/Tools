"""Pre-push hook contracts for changed-file mypy checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _pre_commit_config() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        yaml.safe_load(
            (_REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        ),
    )


def _mypy_hook() -> dict[str, Any]:
    for repo in _pre_commit_config()["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "mypy":
                return cast(dict[str, Any], hook)
    raise AssertionError("mypy pre-push hook is missing")


def _mypy_repo() -> dict[str, Any]:
    """Retrieve the pre-commit repo entry configuring the mypy hook.

    Preconditions:
        The repository root contains a valid `.pre-commit-config.yaml`.

    Postconditions:
        Returns the repository mapping containing the mypy hook.

    Raises:
        AssertionError: If no repo entry contains the mypy hook.
    """
    for repo in _pre_commit_config()["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "mypy":
                return cast(dict[str, Any], repo)
    raise AssertionError("mypy pre-push hook repository is missing")


def test_pre_push_mypy_is_changed_file_delta_scoped() -> None:
    """Pre-push mypy must not fail clean pushes on unrelated imported debt."""
    hook = _mypy_hook()

    assert hook["stages"] == ["pre-push"]
    assert hook["files"] == "^src/"
    assert hook.get("pass_filenames", True) is True
    assert "--follow-imports=skip" in hook["args"]


def test_pre_push_mypy_version_meets_numpy_compat_floor() -> None:
    """Ensure mirrors-mypy rev is >= v1.14.0 to support numpy 2.2+ stubs (#5223)."""
    repo = _mypy_repo()
    rev = str(repo.get("rev", ""))
    assert rev == "v1.15.0", f"mirrors-mypy rev must be v1.15.0, got {rev}"
