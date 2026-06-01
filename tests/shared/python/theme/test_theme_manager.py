from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest


class FakeWindow:
    def __init__(self) -> None:
        self.stylesheets: list[str] = []

    def setStyleSheet(self, stylesheet: str) -> None:
        self.stylesheets.append(stylesheet)


@pytest.fixture
def theme_manager_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from shared.python.theme import theme_manager

    theme_manager.ThemeManager.reset_instance()
    monkeypatch.setattr(
        theme_manager.QStandardPaths,
        "writableLocation",
        lambda _location: str(tmp_path),
    )
    yield theme_manager
    theme_manager.ThemeManager.reset_instance()


def _settings_names() -> tuple[str, str]:
    suffix = uuid4().hex
    return f"ThemeManagerTests-{suffix}", f"App-{suffix}"


def _custom_colors(theme_manager_module) -> dict[str, str]:
    return {
        key: theme_manager_module.BUILTIN_THEMES["Light"][key]
        for key in theme_manager_module.THEME_COLOR_KEYS
    }


def test_singleton_and_convenience_function_share_instance(
    theme_manager_module,
) -> None:
    org, app = _settings_names()

    manager = theme_manager_module.get_theme_manager(
        settings_org=org,
        settings_app=app,
    )
    same_manager = theme_manager_module.ThemeManager.instance(
        settings_org="ignored",
        settings_app="ignored",
    )

    assert same_manager is manager
    assert manager.get_current_theme_name() == "Light"

    theme_manager_module.ThemeManager.reset_instance()
    new_manager = theme_manager_module.ThemeManager.instance(
        settings_org=org,
        settings_app=app,
    )

    assert new_manager is not manager


def test_context_theme_inherits_global_preference(theme_manager_module) -> None:
    org, app = _settings_names()
    global_manager = theme_manager_module.ThemeManager(
        settings_org=org,
        settings_app=app,
    )
    global_manager.change_theme("Dark")

    child_manager = theme_manager_module.ThemeManager(
        app_context="embedded",
        settings_org=org,
        settings_app=app,
    )

    assert child_manager.get_theme_preference() == "Inherit"
    assert child_manager.get_current_theme_name() == "Dark"
    assert child_manager.get_available_themes()[0] == "Inherit"

    child_manager.change_theme("Light")
    assert child_manager.get_theme_preference() == "Light"
    assert child_manager.get_current_theme_name() == "Light"

    child_manager.change_theme("Inherit")
    assert child_manager.get_theme_preference() == "Inherit"
    assert child_manager.get_current_theme_name() == "Dark"


def test_global_manager_rejects_inherit_and_unknown_themes(
    theme_manager_module,
) -> None:
    org, app = _settings_names()
    manager = theme_manager_module.ThemeManager(settings_org=org, settings_app=app)

    manager.change_theme("Inherit")
    assert manager.get_current_theme_name() == "Light"

    manager.change_theme("No Such Theme")
    assert manager.get_current_theme_name() == "Light"


def test_theme_queries_and_stylesheet_fallback(theme_manager_module) -> None:
    org, app = _settings_names()
    manager = theme_manager_module.ThemeManager(settings_org=org, settings_app=app)

    assert "Light" in manager.get_builtin_themes()
    assert manager.get_custom_theme_names() == []
    assert manager.get_theme_colors("Light") == dict(
        theme_manager_module.BUILTIN_THEMES["Light"]
    )
    assert manager.get_theme_definition("No Such Theme") is None
    assert manager.get_theme_stylesheet(
        "No Such Theme"
    ) == manager.get_theme_stylesheet("Light")
    assert manager.get_current_stylesheet() == manager.get_theme_stylesheet("Light")

    with pytest.raises(ValueError, match="theme_name"):
        manager.get_theme_colors(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="theme_name"):
        manager.get_theme_stylesheet(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="theme_name"):
        manager._get_theme_dict(None)  # type: ignore[arg-type]


def test_theme_application_updates_main_and_registered_windows(
    theme_manager_module,
) -> None:
    org, app = _settings_names()
    main_window = FakeWindow()
    registered_window = FakeWindow()
    emitted: list[str] = []
    manager = theme_manager_module.ThemeManager(
        main_window=main_window,
        settings_org=org,
        settings_app=app,
    )
    manager.themeChanged.connect(emitted.append)

    manager.apply_theme_to_window(registered_window)  # type: ignore[arg-type]
    initial_registered_count = len(manager._registered_windows)
    manager.apply_theme_to_window(registered_window)  # type: ignore[arg-type]

    manager.change_theme("Dark")

    assert len(manager._registered_windows) == initial_registered_count
    assert emitted == ["Dark"]
    assert main_window.stylesheets[-1] == manager.get_theme_stylesheet("Dark")
    assert registered_window.stylesheets[-1] == manager.get_theme_stylesheet("Dark")


def test_theme_application_defensive_paths(theme_manager_module) -> None:
    org, app = _settings_names()
    manager = theme_manager_module.ThemeManager(settings_org=org, settings_app=app)
    window = FakeWindow()

    manager.current_theme = "Missing Theme"
    manager.apply_theme()
    assert manager.get_current_theme_name() == "Light"

    manager.apply_theme_by_name(window, "Dark")  # type: ignore[arg-type]
    dark_stylesheet = window.stylesheets[-1]
    manager.apply_theme_by_name(window, "Missing Theme")  # type: ignore[arg-type]
    assert window.stylesheets[-1] == dark_stylesheet

    with pytest.raises(ValueError, match="window"):
        manager.apply_theme_to_window(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="window"):
        manager.apply_theme_by_name(None, "Light")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="window"):
        manager._register_window(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stylesheet"):
        manager._apply_theme_to_registered_windows(None)  # type: ignore[arg-type]


def test_custom_theme_save_load_apply_and_delete(
    theme_manager_module,
    tmp_path: Path,
) -> None:
    org, app = _settings_names()
    colors = _custom_colors(theme_manager_module)
    colors["accent"] = "#abcdef"
    manager = theme_manager_module.ThemeManager(settings_org=org, settings_app=app)

    saved_name = manager.save_custom_theme(
        "  Workbench  ",
        colors,
        apply_immediately=True,
    )

    assert saved_name == "Workbench"
    assert manager.get_current_theme_name() == "Workbench"
    assert manager.get_custom_theme_names() == ["Workbench"]
    assert manager.get_theme_colors("Workbench")["accent"] == "#abcdef"  # type: ignore[index]
    assert (
        json.loads((tmp_path / "user_themes.json").read_text())["Workbench"]["accent"]
        == "#abcdef"
    )

    reloaded = theme_manager_module.ThemeManager(settings_org=org, settings_app=app)
    assert reloaded.get_custom_theme_names() == ["Workbench"]
    assert reloaded.delete_custom_theme("Workbench") is True
    assert reloaded.get_current_theme_name() == "Light"
    assert reloaded.delete_custom_theme("Workbench") is False


def test_save_current_theme_and_custom_theme_validation(
    theme_manager_module,
) -> None:
    org, app = _settings_names()
    manager = theme_manager_module.ThemeManager(settings_org=org, settings_app=app)

    saved_name = manager.save_current_theme_as_custom("Snapshot")

    assert saved_name == "Snapshot"
    assert manager.get_theme_colors("Snapshot") is not None

    with pytest.raises(ValueError, match="empty"):
        manager.save_custom_theme("  ", _custom_colors(theme_manager_module))
    with pytest.raises(ValueError, match="built-in"):
        manager.save_custom_theme("Light", _custom_colors(theme_manager_module))
    with pytest.raises(ValueError, match="theme_name"):
        manager.save_current_theme_as_custom(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="theme_name"):
        manager.delete_custom_theme(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Missing colour values"):
        manager.save_custom_theme("Incomplete", {"accent": "#123456"})
    with pytest.raises(ValueError):
        manager.save_custom_theme(
            "Invalid",
            {**_custom_colors(theme_manager_module), "accent": "not-a-colour"},
        )


def test_load_custom_themes_filters_invalid_payloads(
    theme_manager_module,
    tmp_path: Path,
) -> None:
    org, app = _settings_names()
    valid_colors = _custom_colors(theme_manager_module)
    valid_colors["accent"] = "#123456"
    (tmp_path / "user_themes.json").write_text(
        json.dumps(
            {
                "Valid": valid_colors,
                "Incomplete": {"accent": "#654321"},
                42: valid_colors,
                "NotAMapping": ["#000000"],
            }
        ),
        encoding="utf-8",
    )

    manager = theme_manager_module.ThemeManager(settings_org=org, settings_app=app)

    assert manager.get_custom_theme_names() == ["42", "Valid"]

    (tmp_path / "user_themes.json").write_text("[]", encoding="utf-8")
    manager_with_bad_file = theme_manager_module.ThemeManager(
        settings_org=org,
        settings_app=f"{app}-bad",
    )
    assert manager_with_bad_file.get_custom_theme_names() == []


def test_persist_custom_themes_logs_os_errors(
    theme_manager_module,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    org, app = _settings_names()
    manager = theme_manager_module.ThemeManager(settings_org=org, settings_app=app)
    manager.custom_themes["Broken"] = _custom_colors(theme_manager_module)

    def raise_os_error() -> Path:
        raise OSError("cannot create theme path")

    monkeypatch.setattr(manager, "_get_custom_theme_path", raise_os_error)

    manager._persist_custom_themes()

    assert "Failed to persist custom themes" in caplog.text
