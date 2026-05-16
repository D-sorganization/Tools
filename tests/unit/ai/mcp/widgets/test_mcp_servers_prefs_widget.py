"""Tests for the shared McpServersPrefsWidget.

Architecture: this widget is the canonical home for the MCP server
preferences UI. Both UpstreamDrift and Gasification_Model embed it in
their preferences dialogs — neither owns a copy.

The widget is Qt-based so tests run under a QApplication fixture.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.ai.mcp.contracts import (  # noqa: E402
    McpServerConfig,
    McpTransport,
)
from src.shared.python.ai.mcp.widgets.mcp_servers_prefs_widget import (  # noqa: E402
    McpServersPrefsWidget,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture()
def tmp_config_path(tmp_path: Path) -> Path:
    return tmp_path / "mcp_servers.json"


@pytest.mark.unit
class TestMcpServersPrefsWidgetConstruction:
    def test_construct_empty(self, qapp: QApplication, tmp_config_path: Path) -> None:
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        assert widget.server_count == 0

    def test_construct_loads_existing(
        self, qapp: QApplication, tmp_config_path: Path
    ) -> None:
        # Seed the config file with one server.
        from src.shared.python.ai.mcp.config_writer import write

        cfg = McpServerConfig(
            name="notebooklm",
            transport=McpTransport.STDIO,
            command="python",
            args=["-m", "notebooklm_mcp"],
        )
        write([cfg], path=tmp_config_path)
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        assert widget.server_count == 1
        assert widget.servers[0].name == "notebooklm"


@pytest.mark.unit
class TestMcpServersPrefsWidgetCrud:
    def test_add_server_programmatic(
        self, qapp: QApplication, tmp_config_path: Path
    ) -> None:
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        cfg = McpServerConfig(name="srv1", command="echo")
        widget.add_server(cfg)
        assert widget.server_count == 1

    def test_add_server_rejects_non_config(
        self, qapp: QApplication, tmp_config_path: Path
    ) -> None:
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        with pytest.raises(TypeError):
            widget.add_server("not-a-config")  # type: ignore[arg-type]

    def test_remove_server_by_name(
        self, qapp: QApplication, tmp_config_path: Path
    ) -> None:
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        cfg = McpServerConfig(name="srv1", command="echo")
        widget.add_server(cfg)
        assert widget.remove_server("srv1") is True
        assert widget.server_count == 0

    def test_remove_server_missing_returns_false(
        self, qapp: QApplication, tmp_config_path: Path
    ) -> None:
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        assert widget.remove_server("nope") is False

    def test_persist_writes_file(
        self, qapp: QApplication, tmp_config_path: Path
    ) -> None:
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        cfg = McpServerConfig(name="srv1", command="echo")
        widget.add_server(cfg)
        out = widget.persist()
        assert out == tmp_config_path
        assert tmp_config_path.exists()
        assert "srv1" in tmp_config_path.read_text(encoding="utf-8")


@pytest.mark.unit
class TestMcpServersPrefsWidgetPresets:
    def test_apply_preset_adds_entry(
        self, qapp: QApplication, tmp_config_path: Path
    ) -> None:
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        # Use a fake preset that we register against the widget.
        preset = McpServerConfig(name="preset-srv", command="echo", args=["hi"])
        widget.apply_preset(preset)
        assert widget.server_count == 1
        assert widget.servers[0].name == "preset-srv"

    def test_apply_preset_duplicate_name_skipped(
        self, qapp: QApplication, tmp_config_path: Path
    ) -> None:
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        preset = McpServerConfig(name="preset-srv", command="echo")
        widget.apply_preset(preset)
        widget.apply_preset(preset)  # second call must be a no-op
        assert widget.server_count == 1


@pytest.mark.unit
class TestMcpServersPrefsWidgetImport:
    def test_import_from_claude_desktop_no_op_when_missing(
        self,
        qapp: QApplication,
        tmp_config_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.shared.python.ai.mcp.widgets import mcp_servers_prefs_widget as mod

        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        # Force the discover hook to return an empty list.
        monkeypatch.setattr(mod, "_discover_claude_desktop_servers", lambda: [])
        imported = widget.import_from_claude_desktop()
        assert imported == 0
        assert widget.server_count == 0

    def test_import_from_claude_desktop_adds_entries(
        self,
        qapp: QApplication,
        tmp_config_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.shared.python.ai.mcp.widgets import mcp_servers_prefs_widget as mod

        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        servers = [
            McpServerConfig(name="cd-srv-1", command="echo"),
            McpServerConfig(name="cd-srv-2", command="echo"),
        ]
        monkeypatch.setattr(mod, "_discover_claude_desktop_servers", lambda: servers)
        imported = widget.import_from_claude_desktop()
        assert imported == 2
        assert widget.server_count == 2

    def test_import_from_claude_desktop_skips_duplicates(
        self,
        qapp: QApplication,
        tmp_config_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.shared.python.ai.mcp.widgets import mcp_servers_prefs_widget as mod

        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        widget.add_server(McpServerConfig(name="cd-srv-1", command="echo"))
        servers = [
            McpServerConfig(name="cd-srv-1", command="echo"),
            McpServerConfig(name="cd-srv-2", command="echo"),
        ]
        monkeypatch.setattr(mod, "_discover_claude_desktop_servers", lambda: servers)
        imported = widget.import_from_claude_desktop()
        # Only cd-srv-2 is new.
        assert imported == 1
        assert widget.server_count == 2


@pytest.mark.unit
class TestMcpServersPrefsWidgetSignals:
    def test_servers_changed_signal_on_add(
        self, qapp: QApplication, tmp_config_path: Path
    ) -> None:
        widget = McpServersPrefsWidget(config_path=tmp_config_path)
        spy = MagicMock()
        widget.servers_changed.connect(spy)
        widget.add_server(McpServerConfig(name="s", command="echo"))
        assert spy.called
