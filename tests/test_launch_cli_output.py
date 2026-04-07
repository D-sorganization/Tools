from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_launch_module():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("tools_repo_launch", repo_root / "launch.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRegistry:
    def __init__(self, tools: list[SimpleNamespace]) -> None:
        self._tools = tools

    def list_categories(self) -> list[str]:
        return sorted({tool.category for tool in self._tools})

    def list_tools(self, category: str | None = None) -> list[SimpleNamespace]:
        if category is None:
            return list(self._tools)
        return [tool for tool in self._tools if tool.category == category]

    def get(self, tool_name: str):
        for tool in self._tools:
            if tool.tool_name == tool_name:
                return tool
        return None


def test_list_tools_writes_cli_output_without_print(monkeypatch, capsys) -> None:
    launch = _load_launch_module()
    fake_tools = [
        SimpleNamespace(
            category="Calculators",
            tool_name="pressure_drop_calculator",
            display_name="Pressure Drop Calculator",
        )
    ]
    registry = _FakeRegistry(fake_tools)

    monkeypatch.setattr(launch, "discover_all_tools", lambda: 1)
    monkeypatch.setattr(launch, "get_registry", lambda: registry)

    launch.list_tools()

    output = capsys.readouterr().out
    assert "Discovered 1 tool registrations." in output
    assert "[Calculators]" in output
    assert "pressure_drop_calculator" in output


def test_launch_tool_reports_missing_tool_via_stdout(monkeypatch, capsys) -> None:
    launch = _load_launch_module()
    registry = _FakeRegistry([])

    monkeypatch.setattr(launch, "discover_all_tools", lambda: 0)
    monkeypatch.setattr(launch, "get_registry", lambda: registry)

    result = launch.launch_tool("missing-tool")

    output = capsys.readouterr().out
    assert result == 1
    assert "Tool 'missing-tool' not found." in output
    assert "Use --list to see all available tools." in output


def test_launch_tool_reports_ambiguous_match_via_stdout(monkeypatch, capsys) -> None:
    launch = _load_launch_module()
    fake_tools = [
        SimpleNamespace(
            category="Calculators",
            tool_name="pressure_drop_calculator",
            display_name="Pressure Drop Calculator",
        ),
        SimpleNamespace(
            category="Calculators",
            tool_name="pressure_drop_analyzer",
            display_name="Pressure Drop Analyzer",
        ),
    ]
    registry = _FakeRegistry(fake_tools)

    monkeypatch.setattr(launch, "discover_all_tools", lambda: len(fake_tools))
    monkeypatch.setattr(launch, "get_registry", lambda: registry)

    result = launch.launch_tool("pressure")

    output = capsys.readouterr().out
    assert result == 1
    assert "Ambiguous tool name 'pressure'. Matches:" in output
    assert "pressure_drop_calculator" in output
    assert "pressure_drop_analyzer" in output
