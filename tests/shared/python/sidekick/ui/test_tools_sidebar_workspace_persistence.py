"""Unit tests for ``tools_sidebar.workspace_persistence``.

Qt-free JSON persistence layer for Sidekick workspace registries. Tests cover
the save/load round-trip plus every path-validation and payload-validation
guard. No QApplication required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry
from sidekick.ui.tools_sidebar.workspace_persistence import (
    CALCULATOR_WORKSPACE_FORMAT_VERSION,
    load_workspace_registry,
    save_workspace_registry,
    validate_calculator_workspace_path,
)

SCOPE = "calculator"


# ---------------------------------------------------------------------------
# validate_calculator_workspace_path
# ---------------------------------------------------------------------------


def test_validate_path_accepts_json(tmp_path: Path) -> None:
    target = tmp_path / "ws.json"
    assert validate_calculator_workspace_path(target) == target


def test_validate_path_none_raises() -> None:
    with pytest.raises(ValueError, match="workspace path is required"):
        validate_calculator_workspace_path(None)  # type: ignore[arg-type]


def test_validate_path_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a file"):
        validate_calculator_workspace_path(tmp_path)


def test_validate_path_non_json_suffix_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".json suffix"):
        validate_calculator_workspace_path(tmp_path / "ws.txt")


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    source = WorkspaceRegistry()
    source.set("alpha", 1)
    source.set("beta", [1, 2, 3])
    target = tmp_path / "ws.json"

    written = save_workspace_registry(source, target, scope=SCOPE)
    assert written == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["version"] == CALCULATOR_WORKSPACE_FORMAT_VERSION
    assert payload["scope"] == SCOPE

    dest = WorkspaceRegistry()
    imported = load_workspace_registry(
        dest,
        target,
        expected_scope=SCOPE,
        replace=True,
        confirm_replace=True,
    )
    names = {var.name for var in imported}
    assert {"alpha", "beta"} <= names
    assert dest.get("alpha") == 1


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "ws.json"
    save_workspace_registry(WorkspaceRegistry(), target, scope=SCOPE)
    assert target.exists()


def test_load_replace_without_confirmation_raises(tmp_path: Path) -> None:
    target = tmp_path / "ws.json"
    save_workspace_registry(WorkspaceRegistry(), target, scope=SCOPE)
    with pytest.raises(PermissionError, match="explicit confirmation"):
        load_workspace_registry(
            WorkspaceRegistry(),
            target,
            expected_scope=SCOPE,
            replace=True,
            confirm_replace=False,
        )


# ---------------------------------------------------------------------------
# payload validation
# ---------------------------------------------------------------------------


def _load(tmp_path: Path, payload: object) -> None:
    target = tmp_path / "ws.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    load_workspace_registry(
        WorkspaceRegistry(),
        target,
        expected_scope=SCOPE,
        replace=False,
        confirm_replace=False,
    )


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    target = tmp_path / "ws.json"
    target.write_text("{not valid", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_workspace_registry(
            WorkspaceRegistry(),
            target,
            expected_scope=SCOPE,
            replace=False,
            confirm_replace=False,
        )


def test_load_non_object_payload_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must contain an object"):
        _load(tmp_path, [1, 2, 3])


def test_load_wrong_version_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported workspace version"):
        _load(tmp_path, {"version": 999, "scope": SCOPE, "variables": []})


def test_load_wrong_scope_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scope must be"):
        _load(
            tmp_path,
            {
                "version": CALCULATOR_WORKSPACE_FORMAT_VERSION,
                "scope": "other",
                "variables": [],
            },
        )


def test_load_variables_not_list_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="variables must be a list"):
        _load(
            tmp_path,
            {
                "version": CALCULATOR_WORKSPACE_FORMAT_VERSION,
                "scope": SCOPE,
                "variables": {},
            },
        )


def test_load_variable_without_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must contain names"):
        _load(
            tmp_path,
            {
                "version": CALCULATOR_WORKSPACE_FORMAT_VERSION,
                "scope": SCOPE,
                "variables": [{"value": 1}],
            },
        )
