"""The root package-lock.json records every npm workspace (#5968).

``npm ci`` refuses to install when a workspace in ``package.json`` is missing
from the lock, so a workspace added without regenerating the lock breaks every
Tauri Build ``Install frontend dependencies`` step.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(relative: str) -> dict:
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_lock_lists_every_package_json_workspace() -> None:
    workspaces = _read_json("package.json")["workspaces"]
    locked = _read_json("package-lock.json")["packages"][""]["workspaces"]

    assert sorted(locked) == sorted(workspaces)


def test_lock_records_each_workspace_name_and_version() -> None:
    lock_packages = _read_json("package-lock.json")["packages"]

    for workspace in _read_json("package.json")["workspaces"]:
        manifest = _read_json(f"{workspace}/package.json")
        locked = lock_packages.get(workspace, {})
        # npm omits ``name`` when it equals the workspace folder name.
        locked_name = locked.get("name", Path(workspace).name if locked else None)
        assert locked_name == manifest["name"], workspace
        assert locked.get("version") == manifest.get("version"), workspace
