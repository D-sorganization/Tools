#!/usr/bin/env python3
"""Tests for state manager module."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from upstream_drift_tools.utils.state_manager import (
    StateManager,
    safe_read_json,
    safe_write_json,
)


def test_safe_read_json_missing_file_returns_default(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert safe_read_json(missing, default={"fallback": True}) == {"fallback": True}


def test_safe_read_json_invalid_data_returns_default(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{this is invalid json")
    assert safe_read_json(invalid, default=[]) == []


def test_safe_write_json_roundtrip(tmp_path: Path) -> None:
    file_path = tmp_path / "nested" / "out.json"
    assert safe_write_json(file_path, {"a": 1, "b": 2}) is True
    assert json.loads(file_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}


def test_safe_write_json_non_serializable_returns_false(tmp_path: Path) -> None:
    file_path = tmp_path / "bad.json"
    assert safe_write_json(file_path, {"x": {1, 2, 3}}) is False


@pytest.fixture
def manager(tmp_path: Path) -> StateManager:
    return StateManager(str(tmp_path / "state_root"))


def test_initializes_required_directories(manager: StateManager) -> None:
    assert manager.states_dir.exists()
    assert manager.sessions_dir.exists()
    assert manager.backups_dir.exists()
    assert manager.exports_dir.exists()


def test_save_and_load_state(manager: StateManager) -> None:
    payload = {"value": 42, "name": "alpha"}
    assert manager.save_state("test_state", payload, description="demo") is True
    assert manager.load_state("test_state") == payload


def test_save_state_creates_backup_on_overwrite(manager: StateManager) -> None:
    assert manager.save_state("duplicate", {"v": 1}) is True
    assert manager.save_state("duplicate", {"v": 2}) is True
    backups = list(manager.backups_dir.glob("duplicate_*.backup"))
    assert backups


def test_delete_state_respects_protection(manager: StateManager) -> None:
    assert manager.save_state("protected_state", {"a": 1}, protected=True) is True
    assert manager.delete_state("protected_state", force=False) is False
    assert manager.delete_state("protected_state", force=True) is True


def test_list_states_uses_cache_and_refreshes(manager: StateManager) -> None:
    manager.save_state("s1", {"a": 1})
    states_first = manager.list_states()
    assert manager._states_index_cache is not None
    states_second = manager.list_states()
    assert states_first == states_second

    manager.save_state("s2", {"b": 2})
    refreshed = manager.list_states()
    assert any(item["name"] == "s2" for item in refreshed)


def test_protect_and_unprotect_update_metadata(manager: StateManager) -> None:
    assert manager.save_state("toggle_state", {"x": 1}) is True
    assert manager.protect_state("toggle_state") is True

    state_file = manager.states_dir / "toggle_state.json"
    protected_data = json.loads(state_file.read_text())
    assert protected_data["metadata"]["protected"] is True

    assert manager.unprotect_state("toggle_state") is True
    unprotected_data = json.loads(state_file.read_text())
    assert unprotected_data["metadata"]["protected"] is False


def test_export_and_import_state(manager: StateManager, tmp_path: Path) -> None:
    assert manager.save_state("export_me", {"k": "v"}) is True
    export_path = manager.export_state("export_me")
    assert export_path is not None

    manager2 = StateManager(str(tmp_path / "state_root_2"))
    assert manager2.import_state(export_path, new_name="imported_name") is True
    assert manager2.load_state("imported_name") == {"k": "v"}


def test_import_state_rejects_duplicate_and_bad_file(
    manager: StateManager, tmp_path: Path
) -> None:
    assert manager.save_state("existing", {"v": 1}) is True

    bad_file = tmp_path / "bad.cestate"
    bad_file.write_text('{"invalid":"format"}', encoding="utf-8")
    assert manager.import_state(str(bad_file)) is False

    export_file = tmp_path / "valid.cestate"
    export_payload = {
        "calculator_version": "2.0",
        "state_name": "existing",
        "state_data": {"v": 2},
    }
    export_file.write_text(json.dumps(export_payload), encoding="utf-8")
    assert manager.import_state(str(export_file)) is False


def test_save_and_load_session(manager: StateManager) -> None:
    session = {"window": "open", "active_tab": 3}
    assert manager.save_session(session) is True
    assert manager.load_session() == session


def test_sanitize_filename_and_validate_state(manager: StateManager) -> None:
    assert manager._sanitize_filename('bad<>:"/\\|?*name') == "bad_________name"

    valid = {"metadata": {"name": "ok"}, "data": {"x": 1}}
    invalid = {"metadata": "bad", "data": {"x": 1}}
    assert manager._validate_state(valid) is True
    assert manager._validate_state(invalid) is False


def test_json_serializer_supports_datetime_and_path(manager: StateManager) -> None:
    dt = datetime(2026, 1, 1, 12, 0, 0)
    path = Path("/tmp/demo")
    assert manager._json_serializer(dt).startswith("2026-01-01T12:00:00")
    assert manager._json_serializer(path) == "/tmp/demo"


def test_cleanup_old_backups_removes_expired_files(manager: StateManager) -> None:
    old_backup = manager.backups_dir / "state_20000101_000000.backup"
    old_backup.write_text("old", encoding="utf-8")
    very_old = datetime.now() - timedelta(days=365)
    old_ts = very_old.timestamp()
    old_backup.touch()
    import os

    os.utime(old_backup, (old_ts, old_ts))

    manager.cleanup_old_backups(max_age_days=30)
    assert not old_backup.exists()
