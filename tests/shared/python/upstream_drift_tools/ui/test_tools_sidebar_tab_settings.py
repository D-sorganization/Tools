"""Tests for Sidekick per-tab settings contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_sidebar_tab_definition_accepts_settings_metadata() -> None:
    from upstream_drift_tools.ui.tools_sidebar import (
        SidebarState,
        SidebarTabSettingsDescriptor,
        SidebarTabSettingsSchema,
        SidebarTabSettingsStore,
    )
    from upstream_drift_tools.ui.tools_sidebar.sidebar import SidebarTabDefinition

    plain = SidebarTabDefinition("plain", "Plain", lambda sidebar: None)
    schema = SidebarTabSettingsSchema(
        version=2,
        defaults={"enabled": True, "limit": 10},
        allowed_keys=frozenset({"enabled", "limit"}),
    )
    configured = SidebarTabDefinition(
        "configured",
        "Configured",
        lambda sidebar: None,
        settings=SidebarTabSettingsDescriptor(schema=schema),
    )

    store = SidebarTabSettingsStore([plain, configured], SidebarState())

    assert plain.settings is None
    assert store.settings_for("plain") == {"schema_version": 1, "values": {}}
    assert store.settings_for("configured") == {
        "schema_version": 2,
        "values": {"enabled": True, "limit": 10},
    }


def test_tab_settings_are_persisted_by_tab_instance() -> None:
    from upstream_drift_tools.ui.tools_sidebar import (
        SidebarState,
        SidebarTabSettingsDescriptor,
        SidebarTabSettingsSchema,
        SidebarTabSettingsStore,
    )
    from upstream_drift_tools.ui.tools_sidebar.sidebar import SidebarTabDefinition

    schema = SidebarTabSettingsSchema(
        version=1,
        defaults={"mode": "default", "shared": False},
        allowed_keys=frozenset({"mode", "shared"}),
    )
    source = SidebarTabDefinition(
        "notes",
        "Notes",
        lambda sidebar: None,
        duplicate_enabled=True,
        settings=SidebarTabSettingsDescriptor(schema=schema),
    )
    duplicate = SidebarTabDefinition(
        "notes#1",
        "Notes 2",
        lambda sidebar: None,
        duplicate_enabled=True,
        settings=SidebarTabSettingsDescriptor(schema=schema),
    )
    state = SidebarState(
        tab_settings={
            "notes": {"schema_version": 1, "values": {"mode": "source"}},
            "notes#1": {"schema_version": 1, "values": {"mode": "duplicate"}},
            "missing": {"schema_version": 1, "values": {"mode": "stale"}},
            "notes#2": {"schema_version": 99, "values": {"mode": "future"}},
            "bad": ["not", "a", "mapping"],
        }
    )

    store = SidebarTabSettingsStore([source, duplicate], state)

    assert store.settings_for("notes")["values"]["mode"] == "source"
    assert store.settings_for("notes#1")["values"]["mode"] == "duplicate"
    assert store.materialized_settings() == {
        "notes": {"schema_version": 1, "values": {"mode": "source", "shared": False}},
        "notes#1": {
            "schema_version": 1,
            "values": {"mode": "duplicate", "shared": False},
        },
    }
    assert store.raw_settings()["missing"]["values"]["mode"] == "stale"

    with pytest.raises(KeyError):
        store.update_settings("missing", {"mode": "x"})

    store.update_settings("notes#1", {"mode": "custom", "shared": True})

    assert store.settings_for("notes")["values"]["mode"] == "source"
    assert store.settings_for("notes#1") == {
        "schema_version": 1,
        "values": {"mode": "custom", "shared": True},
    }

    with pytest.raises(ValueError):
        store.update_settings("notes#1", {"unsupported": True})


def test_tab_settings_survive_sidebar_save_and_load(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import (
        SidebarState,
        SidebarTabDefinition,
        SidebarTabSettingsDescriptor,
        SidebarTabSettingsSchema,
        UnifiedToolsSidebar,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    schema = SidebarTabSettingsSchema(
        version=1,
        defaults={"wrap": True},
        allowed_keys=frozenset({"wrap"}),
    )
    definitions = [
        SidebarTabDefinition(
            "notes",
            "Notes",
            lambda sidebar: QtWidgets.QLabel("notes", sidebar),
            duplicate_enabled=True,
            settings=SidebarTabSettingsDescriptor(schema=schema),
        )
    ]
    sidebar = UnifiedToolsSidebar(project_root=tmp_path, tab_definitions=definitions)
    duplicate_id = sidebar.duplicate_tab("notes")

    assert duplicate_id == "notes#1"
    sidebar.update_tab_settings(duplicate_id, {"wrap": False})
    path = tmp_path / "sidekick.json"
    saved = sidebar.save_state(path)
    reloaded = SidebarState.load_json(path)

    assert saved.tab_settings["notes"]["values"] == {"wrap": True}
    assert saved.tab_settings["notes#1"]["values"] == {"wrap": False}
    assert reloaded.tab_settings == saved.tab_settings


def test_selected_tab_settings_gear_opens_panel(tmp_path: Path) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import (
        SidebarTabDefinition,
        SidebarTabSettingsDescriptor,
        SidebarTabSettingsSchema,
        UnifiedToolsSidebar,
    )
    from upstream_drift_tools.ui.tools_sidebar.tab_settings_panel import (
        SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    widget_seen: list[dict[str, object]] = []

    def build_panel(sidebar: UnifiedToolsSidebar, tab_id: str) -> QtWidgets.QWidget:
        widget_seen.append(sidebar.tab_settings(tab_id))
        return QtWidgets.QLabel(f"settings:{tab_id}", sidebar)

    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        tab_definitions=[
            SidebarTabDefinition(
                "scratch",
                "Scratch",
                lambda sidebar: QtWidgets.QLabel("scratch", sidebar),
                settings=SidebarTabSettingsDescriptor(
                    schema=SidebarTabSettingsSchema(
                        version=1,
                        defaults={"enabled": True},
                        allowed_keys=frozenset({"enabled"}),
                    ),
                    widget_factory=build_panel,
                ),
            )
        ],
    )

    button = sidebar.findChild(
        QtWidgets.QToolButton,
        SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME,
    )

    assert button is not None
    assert button.isEnabled() is True

    sidebar.open_active_tab_settings()

    assert widget_seen == [{"schema_version": 1, "values": {"enabled": True}}]
    assert sidebar._settings_dialog is not None
    assert "Scratch Settings" in sidebar._settings_dialog.windowTitle()


def test_tab_settings_backend_imports_remain_qt_lazy() -> None:
    script = """
import sys
from upstream_drift_tools.ui.tools_sidebar import (
    SidebarState,
    SidebarTabSettingsDescriptor,
    SidebarTabSettingsSchema,
    SidebarTabSettingsStore,
)

schema = SidebarTabSettingsSchema(version=1, defaults={"enabled": True})
state = SidebarState()
store = SidebarTabSettingsStore([], state)
assert store.materialized_settings() == {}
assert SidebarTabSettingsDescriptor(schema=schema).schema.version == 1
loaded = [
    name for name in sys.modules
    if name.partition(".")[0] in {"PyQt6", "PySide6", "PyQt5", "PySide2"}
]
assert loaded == [], loaded
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src/shared/python").resolve())
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
