# ruff: noqa: E501
"""TDD tests for MCP GUI integration — status indicator and settings tab.

These tests run headless (no display server required) and exercise:

- ``McpStatusIndicator`` — a lightweight status widget that shows connected
  server count and connection health without importing the full panel.
- ``McpServersTab`` — the settings tab for adding/removing MCP server configs.
- Adapter serialization helpers — ensure MCP-sourced ``McpToolDescriptor``
  round-trips cleanly into the Anthropic, OpenAI, and generic JSON formats
  via the pool's ``_serialize_tool_for_provider`` helper.

All GUI-dependent tests are skipped when PyQt6 is unavailable so the MCP
unit suite stays green in CI environments without a display.
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Now we can import MCP contracts and pool helpers (no Qt needed)
# ---------------------------------------------------------------------------
from src.shared.python.ai.mcp.contracts import (  # noqa: E402
    McpServerConfig,
    McpToolDescriptor,
)
from src.shared.python.ai.mcp.pool import _serialize_tool_for_provider  # noqa: E402

_HAS_PYQT6 = pytest.importorskip("PyQt6.QtCore", reason="PyQt6 required") is not None

# ---------------------------------------------------------------------------
# Non-GUI tests: adapter serialization round-trips
# These always run even without a display.
# ---------------------------------------------------------------------------


class TestAdapterSerialization:
    """Verify that MCP tools serialize correctly for each provider."""

    def _make_tool(self) -> McpToolDescriptor:
        return McpToolDescriptor(
            name="search_notebook",
            description="Search within a notebook",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    def test_serialize_registry_format(self) -> None:
        tool = self._make_tool()
        result = _serialize_tool_for_provider(tool, "notebooklm", provider="registry")
        assert result["namespaced_name"] == "notebooklm:search_notebook"
        assert result["source"] == "mcp:notebooklm"
        assert result["description"] == "Search within a notebook"
        assert "openai" not in result
        assert "anthropic" not in result

    def test_serialize_openai_format(self) -> None:
        tool = self._make_tool()
        result = _serialize_tool_for_provider(tool, "notebooklm", provider="openai")
        assert "openai" in result
        openai_def = result["openai"]
        assert openai_def["type"] == "function"
        fn = openai_def["function"]
        assert fn["name"] == "notebooklm:search_notebook"
        assert fn["description"] == "Search within a notebook"
        assert fn["parameters"]["type"] == "object"

    def test_serialize_anthropic_format(self) -> None:
        tool = self._make_tool()
        result = _serialize_tool_for_provider(tool, "notebooklm", provider="anthropic")
        assert "anthropic" in result
        ant_def = result["anthropic"]
        assert ant_def["name"] == "notebooklm:search_notebook"
        assert ant_def["description"] == "Search within a notebook"
        assert "input_schema" in ant_def

    def test_namespacing_prevents_collision(self) -> None:
        tool = McpToolDescriptor(name="search", description="d", input_schema={})
        r1 = _serialize_tool_for_provider(tool, "server_a")
        r2 = _serialize_tool_for_provider(tool, "server_b")
        assert r1["namespaced_name"] != r2["namespaced_name"]
        assert r1["source"] != r2["source"]

    def test_source_tag_format(self) -> None:
        tool = McpToolDescriptor(name="t", description="d")
        result = _serialize_tool_for_provider(tool, "my_server")
        assert result["source"] == "mcp:my_server"

    def test_input_schema_preserved(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string"},
                "query": {"type": "string"},
            },
            "required": ["notebook_id", "query"],
        }
        tool = McpToolDescriptor(name="t", description="d", input_schema=schema)
        result = _serialize_tool_for_provider(tool, "nb", provider="anthropic")
        assert result["anthropic"]["input_schema"] == schema


# ---------------------------------------------------------------------------
# GUI tests: McpStatusIndicator
# These require PyQt6 and are skipped otherwise.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PYQT6, reason="PyQt6 required")
class TestMcpStatusIndicator:
    """McpStatusIndicator shows server connection health."""

    def _get_indicator_class(self) -> Any:
        """Import McpStatusIndicator lazily so non-Qt envs can skip."""
        from src.shared.python.ai.mcp.gui import (
            McpStatusIndicator,  # type: ignore[import]
        )

        return McpStatusIndicator

    def test_indicator_shows_disconnected_by_default(self, qtbot: Any) -> None:  # type: ignore[name-defined]
        McpStatusIndicator = self._get_indicator_class()
        widget = McpStatusIndicator()
        qtbot.addWidget(widget)
        assert widget.server_count == 0
        assert "disconnected" in widget.status_text.lower() or widget.server_count == 0

    def test_indicator_updates_on_pool_refresh(self, qtbot: Any) -> None:  # type: ignore[name-defined]
        McpStatusIndicator = self._get_indicator_class()
        widget = McpStatusIndicator()
        qtbot.addWidget(widget)
        widget.update_status(connected_count=2, total_count=3)
        assert widget.server_count == 2

    def test_indicator_accessible_text(self, qtbot: Any) -> None:  # type: ignore[name-defined]
        McpStatusIndicator = self._get_indicator_class()
        widget = McpStatusIndicator()
        qtbot.addWidget(widget)
        widget.update_status(connected_count=1, total_count=1)
        text = widget.status_text
        assert "1" in text


# ---------------------------------------------------------------------------
# GUI tests: McpServersTab (settings dialog)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PYQT6, reason="PyQt6 required")
class TestMcpServersTab:
    """McpServersTab allows adding/removing MCP server configs."""

    def _get_tab_class(self) -> Any:
        from src.shared.python.ai.mcp.gui import McpServersTab  # type: ignore[import]

        return McpServersTab

    def test_tab_starts_empty(self, qtbot: Any) -> None:  # type: ignore[name-defined]
        McpServersTab = self._get_tab_class()
        tab = McpServersTab()
        qtbot.addWidget(tab)
        assert tab.server_count == 0

    def test_add_stdio_server(self, qtbot: Any) -> None:  # type: ignore[name-defined]
        McpServersTab = self._get_tab_class()
        tab = McpServersTab()
        qtbot.addWidget(tab)
        cfg = McpServerConfig(name="notebooklm", transport="stdio", command="uvx")
        tab.add_server(cfg)
        assert tab.server_count == 1

    def test_remove_server(self, qtbot: Any) -> None:  # type: ignore[name-defined]
        McpServersTab = self._get_tab_class()
        tab = McpServersTab()
        qtbot.addWidget(tab)
        cfg = McpServerConfig(name="notebooklm", transport="stdio", command="uvx")
        tab.add_server(cfg)
        tab.remove_server("notebooklm")
        assert tab.server_count == 0

    def test_get_configs_returns_all_servers(self, qtbot: Any) -> None:  # type: ignore[name-defined]
        McpServersTab = self._get_tab_class()
        tab = McpServersTab()
        qtbot.addWidget(tab)
        cfg1 = McpServerConfig(name="nb", transport="stdio", command="uvx")
        cfg2 = McpServerConfig(
            name="remote", transport="http", url="https://example.com/mcp"
        )
        tab.add_server(cfg1)
        tab.add_server(cfg2)
        configs = tab.get_configs()
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert names == {"nb", "remote"}
