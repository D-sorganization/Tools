"""Tests for the Sidekick chat MemoryPanel widget (Tools issue #2688).

The chat dock already surfaces history (HistorySidebar) and settings
(provider/model dropdowns) but had no UI for managing persistent
memory. MemoryPanel closes that gap by exposing the existing
:class:`MemoryManager` API as a Qt widget.

These tests are deliberately Qt-instance-free: they bypass
``QWidget.__init__`` so they run headless on CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PyQt6.QtWidgets")


# ─────────────────────── helpers ──────────────────────────────────


def _build_panel(tmp_path: Path):
    """Construct a MemoryPanel bound to a real on-disk MemoryManager."""
    from chat.memory_panel import MemoryPanel
    from src.shared.python.ai.memory_manager import MemoryManager

    manager = MemoryManager(storage_dir=tmp_path)
    with patch(
        "chat.memory_panel.QWidget.__init__",
        return_value=None,
    ):
        panel = MemoryPanel.__new__(MemoryPanel)
        panel._manager = manager
        panel._preferences_view = MagicMock()
        panel._memories_view = MagicMock()
        panel._status_label = MagicMock()
        panel._pref_key_edit = MagicMock()
        panel._pref_value_edit = MagicMock()
    return panel, manager


# ─────────────────────── DbC preconditions ────────────────────────


class TestMemoryPanelPreconditions:
    def test_save_requires_manager(self) -> None:
        """Calling save() without a MemoryManager raises ValueError."""
        from chat.memory_panel import MemoryPanel

        with patch(
            "chat.memory_panel.QWidget.__init__",
            return_value=None,
        ):
            panel = MemoryPanel.__new__(MemoryPanel)
            panel._manager = None
            panel._pref_key_edit = MagicMock()
            panel._pref_value_edit = MagicMock()
        with pytest.raises(ValueError, match="manager"):
            panel.save_preference()

    def test_constructor_rejects_none_manager(self) -> None:
        from chat.memory_panel import MemoryPanel

        with pytest.raises(ValueError, match="manager"):
            MemoryPanel(manager=None)


# ─────────────────────── load / refresh ───────────────────────────


class TestMemoryPanelLoad:
    def test_refresh_reads_preferences_from_manager(self, tmp_path: Path) -> None:
        panel, manager = _build_panel(tmp_path)
        manager.set_preference("style", "concise")
        manager.set_preference("language", "english")
        panel.refresh()
        panel._preferences_view.setPlainText.assert_called_once()
        rendered = panel._preferences_view.setPlainText.call_args[0][0]
        assert "style" in rendered
        assert "concise" in rendered
        assert "language" in rendered

    def test_refresh_reads_memories_from_manager(self, tmp_path: Path) -> None:
        from src.shared.python.ai.memory_manager import MemoryCandidate

        panel, manager = _build_panel(tmp_path)
        manager.add_memory(
            MemoryCandidate(
                kind="preference",
                content="remember to use SI units",
                source="s1:0",
                source_hash="hash-1",
            )
        )
        panel.refresh()
        panel._memories_view.setPlainText.assert_called_once()
        rendered = panel._memories_view.setPlainText.call_args[0][0]
        assert "SI units" in rendered


# ─────────────────────── save preference ──────────────────────────


class TestMemoryPanelSave:
    def test_save_preference_writes_to_manager(self, tmp_path: Path) -> None:
        panel, manager = _build_panel(tmp_path)
        panel._pref_key_edit.text.return_value = "tone"
        panel._pref_value_edit.text.return_value = "formal"
        panel.save_preference()
        assert manager.memory["preferences"]["tone"] == "formal"

    def test_save_preference_rejects_empty_key(self, tmp_path: Path) -> None:
        panel, _ = _build_panel(tmp_path)
        panel._pref_key_edit.text.return_value = "   "
        panel._pref_value_edit.text.return_value = "x"
        with pytest.raises(ValueError):
            panel.save_preference()


# ─────────────────────── clear all ────────────────────────────────


class TestMemoryPanelClearAll:
    def test_clear_all_requires_confirmation(self, tmp_path: Path) -> None:
        panel, manager = _build_panel(tmp_path)
        manager.set_preference("k", "v")
        # Confirmation declined -> nothing changes
        panel.clear_all(confirm=lambda: False)
        assert manager.memory["preferences"] == {"k": "v"}

    def test_clear_all_with_confirmation_wipes_state(self, tmp_path: Path) -> None:
        from src.shared.python.ai.memory_manager import MemoryCandidate

        panel, manager = _build_panel(tmp_path)
        manager.set_preference("k", "v")
        manager.add_memory(
            MemoryCandidate(
                kind="preference",
                content="remember x",
                source="s:0",
                source_hash="h",
            )
        )
        panel.clear_all(confirm=lambda: True)
        assert manager.memory["preferences"] == {}
        assert manager.memory["memories"] == []

    def test_clear_all_default_confirm_is_required(self, tmp_path: Path) -> None:
        """DbC: clear_all() without an explicit confirm callable must raise."""
        panel, _ = _build_panel(tmp_path)
        with pytest.raises(ValueError, match="confirm"):
            panel.clear_all(confirm=None)


# ─────────────────────── export / import round-trip ───────────────


class TestMemoryPanelExportImport:
    def test_export_round_trip(self, tmp_path: Path) -> None:
        from src.shared.python.ai.memory_manager import MemoryCandidate, MemoryManager

        panel, manager = _build_panel(tmp_path)
        manager.set_preference("style", "concise")
        manager.add_memory(
            MemoryCandidate(
                kind="preference",
                content="remember to use SI units",
                source="s:0",
                source_hash="round-trip-hash",
            )
        )

        export_path = tmp_path / "export.json"
        panel.export_to(export_path)
        assert export_path.exists()
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert payload["preferences"]["style"] == "concise"
        assert any(
            m.get("content") == "remember to use SI units"
            for m in payload.get("memories", [])
        )

        # Import into a fresh manager and verify content is preserved.
        fresh_dir = tmp_path / "fresh"
        fresh_dir.mkdir()
        with patch(
            "chat.memory_panel.QWidget.__init__",
            return_value=None,
        ):
            from chat.memory_panel import MemoryPanel

            other = MemoryPanel.__new__(MemoryPanel)
            other._manager = MemoryManager(storage_dir=fresh_dir)
            other._preferences_view = MagicMock()
            other._memories_view = MagicMock()
            other._status_label = MagicMock()
            other._pref_key_edit = MagicMock()
            other._pref_value_edit = MagicMock()
        other.import_from(export_path)
        assert other._manager.memory["preferences"]["style"] == "concise"
        contents = [m.get("content") for m in other._manager.memory["memories"]]
        assert "remember to use SI units" in contents

    def test_import_rejects_missing_file(self, tmp_path: Path) -> None:
        panel, _ = _build_panel(tmp_path)
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            panel.import_from(missing)

    def test_import_rejects_invalid_json(self, tmp_path: Path) -> None:
        panel, _ = _build_panel(tmp_path)
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError):
            panel.import_from(bad)


# ─────────────────────── dock wiring ──────────────────────────────


class TestChatDockMemoryWiring:
    def test_dock_exposes_open_memory_panel_action(self) -> None:
        """The chat dock's Tools menu must include a Memory action."""
        pytest.importorskip("PyQt6.QtWebSockets")
        from chat._chat_dock_widget_qt import ChatDockWidget

        # The action attribute must exist on the class so the menu can
        # be discovered by tests/UI inspectors without instantiating Qt.
        assert hasattr(ChatDockWidget, "open_memory_panel")
