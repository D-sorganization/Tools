from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PyQt6")


def test_chat_accent_color_resolution_contract() -> None:
    from sidekick.ui.tools_sidebar import chat_tab

    class MappingProvider:
        def get_current_colors(self) -> dict[str, str]:
            return {"accent": "#123456"}

    class TokenProvider:
        def tokens(self) -> Any:
            return types.SimpleNamespace(accent="#abcdef")

    class CallableAccentProvider:
        def accent_color(self) -> str:
            return "#654321"

    assert chat_tab._resolve_accent_color(None) == "#FF8800"
    assert chat_tab._resolve_accent_color(MappingProvider()) == "#123456"
    assert chat_tab._resolve_accent_color(TokenProvider()) == "#abcdef"
    assert chat_tab._resolve_accent_color(CallableAccentProvider()) == "#654321"
    assert chat_tab._resolve_accent_color(types.SimpleNamespace(accent_color="#fff000"))


def test_sidebar_workspace_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from sidekick.ui.tools_sidebar.chat_tab import _SidebarWorkspaceAdapter
    from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

    @dataclass(frozen=True)
    class WorkspaceVariableInfo:
        name: str
        dtype: str
        shape: tuple[int, ...] | None
        preview: str

    chat_pkg = types.ModuleType("chat")
    chat_protocol = types.ModuleType("chat._workspace_protocol")
    chat_protocol.WorkspaceVariableInfo = WorkspaceVariableInfo
    monkeypatch.setitem(sys.modules, "chat", chat_pkg)
    monkeypatch.setitem(sys.modules, "chat._workspace_protocol", chat_protocol)

    registry = WorkspaceRegistry()
    registry.set("alpha", [1, 2, 3])
    adapter = _SidebarWorkspaceAdapter(registry)

    described = adapter.describe()
    assert described[0].name == "alpha"
    assert described[0].preview
    assert adapter.read("alpha") == [1, 2, 3]

    adapter.write("beta", 42)
    assert registry.get("beta") == 42
    with pytest.raises(KeyError):
        adapter.read("missing")
    with pytest.raises(TypeError):
        adapter.write(123, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _SidebarWorkspaceAdapter(None)  # type: ignore[arg-type]


def test_chat_plot_sink_activates_hidden_plot_tab() -> None:
    from sidekick.ui.tools_sidebar.chat_tab import _build_sidebar_plot_request_sink
    from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

    class PlotWidget:
        def __init__(self) -> None:
            self.spec: Any = None

        def set_spec(self, spec: Any) -> None:
            self.spec = spec

    class Sidebar:
        def __init__(self) -> None:
            self.registry = WorkspaceRegistry()
            self.registry.set("ys", [1, 2, 3])
            self._tab_widgets: dict[str, Any] = {}

        def set_tab_visible(self, tab_id: str, visible: bool) -> None:
            assert visible is True
            self._tab_widgets[tab_id] = PlotWidget()

    sidebar = Sidebar()
    sink = _build_sidebar_plot_request_sink(sidebar)
    assert sink is not None

    sink({"source": "workspace_result", "y_ref": "ys", "title": "Series"})

    widget = sidebar._tab_widgets["calculator_plot"]
    assert widget.spec.title == "Series"
    assert widget.spec.series[0].y == [1.0, 2.0, 3.0]


def test_chat_status_tab_formats_error_and_retry(qt_app: Any, qtbot: Any) -> None:
    from sidekick.ui.tools_sidebar import chat_tab

    sidebar = chat_tab.QtWidgets.QWidget()
    qtbot.addWidget(sidebar)
    sidebar._chat_dock_import_error = RuntimeError("chat missing")

    widget = chat_tab._build_chat_status_tab(sidebar)
    qtbot.addWidget(widget)

    error_view = widget.findChild(
        chat_tab.QtWidgets.QPlainTextEdit, "SidekickChatStatusError"
    )
    assert error_view is not None
    assert "chat missing" in error_view.toPlainText()
    assert "RuntimeError" in chat_tab._format_chat_import_error(RuntimeError("boom"))
    assert "Reason unknown" in chat_tab._format_chat_import_error(None)


def test_default_tab_definitions_are_registered_with_settings(tmp_path: Path) -> None:
    from sidekick.ui.tools_sidebar import default_tabs
    from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

    class Sidebar:
        project_root = tmp_path
        registry = WorkspaceRegistry()

    def definition(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return args, kwargs

    definitions = default_tabs.build_default_tab_definitions(Sidebar(), definition)
    ids = [args[0] for args, _kwargs in definitions]

    assert ids[:3] == ["chat", "files", "workspace"]
    assert "calculator_plot" in ids
    assert "reporting" in ids
    chat = definitions[0]
    assert chat[1]["settings"] is default_tabs.CHAT_TAB_SETTINGS
    assert chat[1]["help_metadata"] is not default_tabs.DEFAULT_SIDEBAR_TAB_HELP["chat"]


def test_default_optional_tabs_return_placeholders(
    qt_app: Any,
    qtbot: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sidekick.ui.tools_sidebar import default_tabs

    sidebar = default_tabs.QtWidgets.QWidget()
    qtbot.addWidget(sidebar)
    monkeypatch.setattr(default_tabs, "QT_API", "stub")

    builders = [
        default_tabs.build_unit_converter_tab,
        default_tabs.build_calculator_plot_tab,
        default_tabs.build_rotation_converter_tab,
        default_tabs.build_function_generator_tab,
    ]

    for builder in builders:
        widget = builder(sidebar)
        qtbot.addWidget(widget)
        assert (
            widget.objectName() == default_tabs.theme.SIDEKICK_PLACEHOLDER_OBJECT_NAME
        )
