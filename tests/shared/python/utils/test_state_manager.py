"""Tests for upstream_drift_tools.utils.state_manager module.

Covers:
- safe_read_json / safe_write_json
- StateManager: save_state, load_state, delete_state, list_states
- State protection (protect/unprotect)
- State export/import
- Session save/load
- Filename sanitization
- State validation
"""

from __future__ import annotations

from pathlib import Path

import pytest
from upstream_drift_tools.utils.state_manager import (
    StateManager,
    safe_read_json,
    safe_write_json,
)

# ── safe_read_json / safe_write_json ─────────────────────────────────────


class TestSafeJsonIO:
    """Test JSON read/write utilities."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        data = {"key": "value", "num": 42}
        safe_write_json(path, data)
        result = safe_read_json(path)
        assert result == data

    def test_read_nonexistent_returns_default(self, tmp_path: Path) -> None:
        result = safe_read_json(tmp_path / "missing.json", default={"a": 1})
        assert result == {"a": 1}

    def test_read_corrupt_returns_default(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        result = safe_read_json(path, default=[])
        assert result == []

    def test_write_creates_parents(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "data.json"
        safe_write_json(path, {"nested": True}, create_parents=True)
        assert path.exists()
        assert safe_read_json(path)["nested"] is True

    def test_write_with_indent(self, tmp_path: Path) -> None:
        path = tmp_path / "pretty.json"
        safe_write_json(path, {"a": 1}, indent=4)
        content = path.read_text()
        assert "    " in content  # 4-space indent.

    def test_roundtrip_complex_data(self, tmp_path: Path) -> None:
        path = tmp_path / "complex.json"
        data = {
            "list": [1, 2, 3],
            "nested": {"a": {"b": "c"}},
            "null": None,
            "bool": True,
        }
        safe_write_json(path, data)
        assert safe_read_json(path) == data


# ── StateManager ─────────────────────────────────────────────────────────


class TestStateManager:
    """Test StateManager save/load/delete/list operations."""

    @pytest.fixture()
    def mgr(self, tmp_path: Path) -> StateManager:
        return StateManager(base_directory=str(tmp_path / "states"))

    def test_save_and_load(self, mgr: StateManager) -> None:
        state = {"temperature": 1200.0, "pressure": 101325.0}
        mgr.save_state("test_run", state, description="test")
        loaded = mgr.load_state("test_run")
        assert loaded is not None
        assert loaded["temperature"] == 1200.0

    def test_load_nonexistent_returns_none(self, mgr: StateManager) -> None:
        result = mgr.load_state("nonexistent")
        assert result is None

    def test_list_states_empty(self, mgr: StateManager) -> None:
        states = mgr.list_states()
        assert states == []

    def test_list_states_after_save(self, mgr: StateManager) -> None:
        mgr.save_state("alpha", {"v": 1})
        mgr.save_state("beta", {"v": 2})
        states = mgr.list_states()
        names = [s["name"] for s in states]
        assert "alpha" in names
        assert "beta" in names

    def test_delete_state(self, mgr: StateManager) -> None:
        mgr.save_state("to_delete", {"v": 99})
        assert mgr.load_state("to_delete") is not None
        mgr.delete_state("to_delete")
        assert mgr.load_state("to_delete") is None

    def test_delete_nonexistent_returns_false(self, mgr: StateManager) -> None:
        result = mgr.delete_state("ghost")
        assert result is False

    def test_overwrite_state(self, mgr: StateManager) -> None:
        mgr.save_state("run1", {"version": 1})
        mgr.save_state("run1", {"version": 2})
        loaded = mgr.load_state("run1")
        assert loaded is not None
        assert loaded["version"] == 2


# ── Protection ───────────────────────────────────────────────────────────


class TestStateProtection:
    """Test state protection mechanism."""

    @pytest.fixture()
    def mgr(self, tmp_path: Path) -> StateManager:
        return StateManager(base_directory=str(tmp_path / "states"))

    def test_protect_prevents_delete(self, mgr: StateManager) -> None:
        mgr.save_state("important", {"data": 1}, protected=True)
        result = mgr.delete_state("important")
        assert result is False  # Protected states can't be deleted
        assert mgr.load_state("important") is not None  # Still exists

    def test_force_delete_protected(self, mgr: StateManager) -> None:
        mgr.save_state("important", {"data": 1}, protected=True)
        result = mgr.delete_state("important", force=True)
        assert result is True
        assert mgr.load_state("important") is None

    def test_protect_and_unprotect(self, mgr: StateManager) -> None:
        mgr.save_state("toggle", {"v": 1})
        mgr.protect_state("toggle")
        result = mgr.delete_state("toggle")
        assert result is False  # Protected, so delete fails
        mgr.unprotect_state("toggle")
        result = mgr.delete_state("toggle")
        assert result is True  # Now unprotected, delete succeeds


# ── Session ──────────────────────────────────────────────────────────────


class TestSessionManagement:
    """Test session save/load."""

    @pytest.fixture()
    def mgr(self, tmp_path: Path) -> StateManager:
        return StateManager(base_directory=str(tmp_path / "states"))

    def test_save_and_load_session(self, mgr: StateManager) -> None:
        session = {"last_file": "data.csv", "zoom": 1.5}
        mgr.save_session(session)
        loaded = mgr.load_session()
        assert loaded is not None
        assert loaded["last_file"] == "data.csv"

    def test_load_session_none_when_empty(self, mgr: StateManager) -> None:
        loaded = mgr.load_session()
        assert loaded is None


# ── Export / Import ──────────────────────────────────────────────────────


class TestStateExportImport:
    """Test state export and import."""

    @pytest.fixture()
    def mgr(self, tmp_path: Path) -> StateManager:
        return StateManager(base_directory=str(tmp_path / "states"))

    def test_export_state(self, mgr: StateManager, tmp_path: Path) -> None:
        mgr.save_state("export_me", {"val": 42})
        export_path = tmp_path / "exported.json"
        mgr.export_state("export_me", str(export_path))
        assert export_path.exists()

    def test_import_state(self, mgr: StateManager, tmp_path: Path) -> None:
        mgr.save_state("orig", {"val": 100})
        export_path = tmp_path / "exported.json"
        mgr.export_state("orig", str(export_path))
        mgr.delete_state("orig")
        mgr.import_state(str(export_path), new_name="imported")
        loaded = mgr.load_state("imported")
        assert loaded is not None

    def test_export_nonexistent_returns_none(self, mgr: StateManager) -> None:
        result = mgr.export_state("missing")
        assert result is None


# ── Filename Sanitization ────────────────────────────────────────────────


class TestFilenameSanitization:
    """Test that filenames are properly sanitized."""

    @pytest.fixture()
    def mgr(self, tmp_path: Path) -> StateManager:
        return StateManager(base_directory=str(tmp_path / "states"))

    def test_special_chars_removed(self, mgr: StateManager) -> None:
        mgr.save_state("test/state:name", {"v": 1})
        loaded = mgr.load_state("test/state:name")
        assert loaded is not None

    def test_spaces_handled(self, mgr: StateManager) -> None:
        mgr.save_state("my state name", {"v": 1})
        loaded = mgr.load_state("my state name")
        assert loaded is not None
