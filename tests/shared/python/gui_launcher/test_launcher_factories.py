"""Focused coverage for GUI launcher factory helpers."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import pytest
from gui_launcher.launcher import GUILauncher, GUIType, LaunchConfig
from gui_launcher.launcher_factories import (
    create_launcher,
    generate_launch_script,
    launch_tool_by_name,
    make_launcher,
    make_pyqt6_launcher,
)
from gui_launcher.registry import GUIRegistry, register_gui

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_global_registry() -> Generator[None, None, None]:
    GUIRegistry._instance = None
    yield
    GUIRegistry._instance = None


def _pyqt6_config(tool_name: str = "demo") -> LaunchConfig:
    return LaunchConfig(
        tool_name=tool_name,
        gui_type=GUIType.PYQT6,
        module_path=f"{tool_name}.ui",
        class_name="DemoWindow",
        dependencies=["PyQt6"],
    )


def test_create_launcher_builds_configured_launcher() -> None:
    launcher = create_launcher(
        "demo",
        GUIType.PYQT6,
        module_path="demo.ui",
        class_name="DemoWindow",
        dependencies=["PyQt6"],
        title="Demo Tool",
    )

    assert isinstance(launcher, GUILauncher)
    assert launcher.config.tool_name == "demo"
    assert launcher.config.gui_type is GUIType.PYQT6
    assert launcher.config.module_path == "demo.ui"
    assert launcher.config.class_name == "DemoWindow"
    assert launcher.config.dependencies == ["PyQt6"]
    assert launcher.config.title == "Demo Tool"


def test_create_launcher_rejects_missing_tool_name() -> None:
    with pytest.raises(ValueError, match="tool_name must be provided"):
        create_launcher(None, GUIType.PYQT6)


def test_generate_launch_script_contains_bootstrap_and_make_launcher_call() -> None:
    script = generate_launch_script("demo.gui_registration", "Demo Tool")

    assert script.startswith("#!/usr/bin/env python3\n")
    assert '"""Standalone PyQt6 launcher for Demo Tool."""' in script
    assert "from _bootstrap import bootstrap  # noqa: E402" in script
    assert "bootstrap(__file__)" in script
    assert "from gui_launcher import make_launcher  # noqa: E402" in script
    assert 'sys.exit(make_launcher("demo.gui_registration"))' in script


def test_generate_launch_script_rejects_missing_module() -> None:
    with pytest.raises(ValueError, match="gui_info_module must be provided"):
        generate_launch_script(None, "Demo Tool")


def test_launch_tool_by_name_reports_missing_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_gui(
        "available",
        "Available Tool",
        "Registered for logging",
        {GUIType.PYQT6: _pyqt6_config("available")},
    )

    with caplog.at_level("INFO", logger="gui_launcher.launcher_factories"):
        result = launch_tool_by_name("missing")

    assert result == 1
    assert "Tool 'missing' not found in registry." in caplog.text
    assert "Available tools:" in caplog.text
    assert "available (Available Tool)" in caplog.text


def test_launch_tool_by_name_reports_missing_pyqt6_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_gui(
        "web_only",
        "Web Only",
        "No PyQt6 config",
        {
            GUIType.REACT: LaunchConfig(
                tool_name="web_only",
                gui_type=GUIType.REACT,
                web_path="web",
            )
        },
    )

    with caplog.at_level("INFO", logger="gui_launcher.launcher_factories"):
        result = launch_tool_by_name("web_only")

    assert result == 1
    assert "Tool 'web_only' has no PyQt6 configuration." in caplog.text


def test_launch_tool_by_name_dispatches_pyqt6_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[LaunchConfig] = []
    config = _pyqt6_config("launchable")
    register_gui(
        "launchable",
        "Launchable Tool",
        "Has PyQt6 config",
        {GUIType.PYQT6: config},
    )

    def fake_launch_pyqt6_app(received: LaunchConfig) -> int:
        captured.append(received)
        return 23

    monkeypatch.setattr(
        "gui_launcher.launcher.launch_pyqt6_app",
        fake_launch_pyqt6_app,
    )

    assert launch_tool_by_name("launchable") == 23
    assert captured == [config]


def test_make_launcher_delegates_to_pyqt6_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_make_pyqt6_launcher(module_name: str) -> int:
        calls.append(module_name)
        return 7

    monkeypatch.setattr(
        "gui_launcher.launcher_factories.make_pyqt6_launcher",
        fake_make_pyqt6_launcher,
    )

    assert make_launcher("demo.gui_registration") == 7
    assert calls == ["demo.gui_registration"]


def test_make_pyqt6_launcher_reports_import_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("ERROR", logger="gui_launcher.launcher_factories"):
        result = make_pyqt6_launcher("missing_gui_registration_module")

    assert result == 1
    assert "Failed to import GUI registration module" in caplog.text


def test_make_pyqt6_launcher_reports_missing_gui_info(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module_path = tmp_path / "no_gui_info.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        with caplog.at_level("ERROR", logger="gui_launcher.launcher_factories"):
            result = make_pyqt6_launcher("no_gui_info")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("no_gui_info", None)

    assert result == 1
    assert "does not define a GUI_INFO dict" in caplog.text


def test_make_pyqt6_launcher_launches_gui_info_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("fake_gui_info_module")
    gui_info = {
        "tool_name": "fake",
        "name": "Fake GUI",
        "pyqt6": {
            "module": "fake.ui",
            "class": "FakeWindow",
        },
    }
    module.__dict__["GUI_INFO"] = gui_info
    sys.modules[module.__name__] = module
    captured: list[dict[str, object]] = []

    def fake_launch_from_gui_info(gui_info: dict[str, object]) -> int:
        captured.append(gui_info)
        return 42

    monkeypatch.setattr(
        "gui_launcher.launcher.launch_from_gui_info",
        fake_launch_from_gui_info,
    )
    try:
        assert make_pyqt6_launcher(module.__name__) == 42
    finally:
        sys.modules.pop(module.__name__, None)

    assert captured == [gui_info]
