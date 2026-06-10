"""Focused coverage for GUI launcher registry helpers."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import pytest
from gui_launcher.launcher import GUIType, LaunchConfig
from gui_launcher.registry import (
    GUIRegistry,
    _gui_info_to_registration,
    auto_discover_guis,
    get_registry,
    register_gui,
)

from contracts import PreconditionError

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_global_registry() -> Generator[None, None, None]:
    GUIRegistry._instance = None
    yield
    GUIRegistry._instance = None


def _pyqt6_config(tool_name: str = "alpha") -> LaunchConfig:
    return LaunchConfig(
        tool_name=tool_name,
        gui_type=GUIType.PYQT6,
        module_path=f"{tool_name}.ui",
        class_name="MainWindow",
        dependencies=["PyQt6"],
    )


def test_registry_register_get_list_and_clear(caplog: pytest.LogCaptureFixture) -> None:
    registry = GUIRegistry()
    alpha = _pyqt6_config("alpha")
    beta = _pyqt6_config("beta")

    with caplog.at_level(logging.DEBUG, logger="gui_launcher.registry"):
        registry.register(
            "beta",
            "Beta Tool",
            "Beta description",
            {GUIType.PYQT6: beta},
            category="Utilities",
            icon="beta-icon",
            repository="Tools",
        )
        registry.register(
            "alpha",
            "Alpha Tool",
            "Alpha description",
            {GUIType.PYQT6: alpha},
            category="Science",
        )

    assert registry.get("alpha").display_name == "Alpha Tool"
    assert registry.get_config("beta", GUIType.PYQT6) is beta
    assert registry.get_config("missing", GUIType.PYQT6) is None
    assert registry.list_tools() == [registry.get("alpha"), registry.get("beta")]
    assert registry.list_tools(category="Science") == [registry.get("alpha")]
    assert registry.list_categories() == ["Science", "Utilities"]
    assert registry.get_available_gui_types("alpha") == [GUIType.PYQT6]
    assert registry.get_available_gui_types("missing") == []
    assert registry.unregister("missing") is False
    assert registry.unregister("beta") is True
    assert "Registered GUI: alpha (1 variants)" in caplog.text

    registry.clear()
    assert registry.list_tools() == []


@pytest.mark.parametrize(
    ("callable_name", "args"),
    [
        ("register", (None, "Name", "Description", {GUIType.PYQT6: _pyqt6_config()})),
        ("register", ("", "Name", "Description", {GUIType.PYQT6: _pyqt6_config()})),
        ("register", ("tool", "", "Description", {GUIType.PYQT6: _pyqt6_config()})),
        ("register", ("tool", "Name", "Description", {})),
        (
            "register",
            ("tool", "Name", "Description", {GUIType.PYQT6: _pyqt6_config()}, ""),
        ),
        ("unregister", (None,)),
        ("unregister", ("",)),
        ("get", ("",)),
        ("get_config", ("tool", "pyqt6")),
    ],
)
def test_registry_rejects_invalid_inputs(
    callable_name: str, args: tuple[object, ...]
) -> None:
    registry = GUIRegistry()

    with pytest.raises((PreconditionError, ValueError)):
        getattr(registry, callable_name)(*args)


def test_registry_get_and_get_config_reject_none_tool_name() -> None:
    registry = GUIRegistry()

    with pytest.raises(ValueError, match="tool_name must be provided"):
        registry.get(None)
    with pytest.raises(ValueError, match="tool_name must be provided"):
        registry.get_config(None, GUIType.PYQT6)


def test_global_registry_helpers_register_with_singleton() -> None:
    config = _pyqt6_config("global")

    assert get_registry() is get_registry()
    register_gui(
        "global",
        "Global Tool",
        "Registered through helper",
        {GUIType.PYQT6: config},
        category="Helpers",
    )

    registration = get_registry().get("global")
    assert registration is not None
    assert registration.display_name == "Global Tool"
    assert registration.gui_configs[GUIType.PYQT6] is config

    with pytest.raises(ValueError, match="tool_name must be provided"):
        register_gui(None, "Name", "Description", {GUIType.PYQT6: config})


def test_gui_info_to_registration_converts_pyqt6_config() -> None:
    _gui_info_to_registration(
        {
            "tool_name": "manifest_tool",
            "name": "Manifest Tool",
            "description": "From GUI_INFO",
            "category": "Manifest",
            "icon": "manifest-icon",
            "pyqt6": {
                "module": "manifest_tool.ui",
                "class": "ManifestWindow",
                "dependencies": ["PyQt6", "numpy"],
                "settings_app": "ManifestTool",
                "min_size": [900, 600],
            },
        }
    )

    registration = get_registry().get("manifest_tool")
    assert registration is not None
    config = registration.gui_configs[GUIType.PYQT6]
    assert registration.display_name == "Manifest Tool"
    assert registration.category == "Manifest"
    assert registration.icon == "manifest-icon"
    assert config.module_path == "manifest_tool.ui"
    assert config.class_name == "ManifestWindow"
    assert config.dependencies == ["PyQt6", "numpy"]
    assert config.title == "Manifest Tool"
    assert config.settings_app == "ManifestTool"
    assert config.min_size == (900, 600)


def test_gui_info_without_gui_config_does_not_register() -> None:
    _gui_info_to_registration({"tool_name": "metadata_only", "name": "Metadata Only"})

    assert get_registry().get("metadata_only") is None


def test_auto_discover_guis_registers_gui_info_modules(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    package = tmp_path / "tool_pkg"
    package.mkdir()
    (package / "gui_registration.py").write_text(
        """
GUI_INFO = {
    "tool_name": "discovered",
    "name": "Discovered Tool",
    "description": "Loaded from disk",
    "category": "Discovery",
    "pyqt6": {
        "module": "discovered.ui",
        "class": "DiscoveredWindow",
    },
}
""",
        encoding="utf-8",
    )

    with caplog.at_level(logging.DEBUG, logger="gui_launcher.registry"):
        count = auto_discover_guis([tmp_path])

    assert count == 1
    assert get_registry().get("discovered").display_name == "Discovered Tool"
    assert "Loaded GUI registration from:" in caplog.text


def test_auto_discover_guis_does_not_count_modules_without_gui_info(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A module without a valid GUI_INFO is not a successful registration.

    The return value reflects GUIs actually registered, not modules merely
    imported, so a registration-less module counts 0 and is logged.
    """
    package = tmp_path / "legacy_tool"
    package.mkdir()
    (package / "gui_registration.py").write_text("VALUE = 1\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="gui_launcher.registry"):
        count = auto_discover_guis([tmp_path])

    assert count == 0
    assert get_registry().list_tools() == []
    assert "missing/invalid" in caplog.text


def test_one_bad_registration_does_not_abort_discovery(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single broken gui_registration.py must not abort discovery for others.

    Regression test for #3275: a non-ImportError raised at import time (here a
    RuntimeError) previously propagated out of the loop and took down discovery
    for every tool. It must instead be skipped with a logged warning while the
    valid registration still succeeds.
    """
    good = tmp_path / "good_tool"
    good.mkdir()
    (good / "gui_registration.py").write_text(
        """
GUI_INFO = {
    "tool_name": "good",
    "name": "Good Tool",
    "description": "Valid registration",
    "category": "Discovery",
    "pyqt6": {"module": "good.ui", "class": "GoodWindow"},
}
""",
        encoding="utf-8",
    )

    bad = tmp_path / "bad_tool"
    bad.mkdir()
    bad_reg = bad / "gui_registration.py"
    bad_reg.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="gui_launcher.registry"):
        count = auto_discover_guis([tmp_path])

    assert count == 1
    assert get_registry().get("good").display_name == "Good Tool"
    assert f"Failed to load GUI registration from {bad_reg}" in caplog.text


def test_auto_discover_guis_skips_missing_paths_and_warns_on_import_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    package = tmp_path / "broken_tool"
    package.mkdir()
    reg_file = package / "gui_registration.py"
    reg_file.write_text(
        "raise ImportError('optional dependency missing')\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="gui_launcher.registry"):
        count = auto_discover_guis([tmp_path / "missing", tmp_path])

    assert count == 0
    assert f"Failed to load GUI registration from {reg_file}" in caplog.text


def test_auto_discover_guis_requires_list() -> None:
    with pytest.raises(PreconditionError, match="search_paths must be a list of Paths"):
        auto_discover_guis(tuple())
