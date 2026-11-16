"""Configuration fallback, input, and platform boundaries using local files."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from utils import config_loader

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("payload", [[], None, 123, "text", False])
def test_non_object_json_uses_an_independent_default_copy(
    payload: Any, tmp_path: Path
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    defaults = {"theme": "dark"}
    loader = config_loader.ConfigLoader(path, defaults)
    loaded = loader.load()
    assert loaded == defaults
    assert loaded is not defaults
    loaded["theme"] = "light"
    assert defaults == {"theme": "dark"}


def test_invalid_json_uses_defaults_and_reports_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")
    assert config_loader.load_config(path, {"fallback": True}) == {"fallback": True}
    assert "Invalid JSON" in caplog.text


def test_nested_get_stops_at_scalar_or_null_and_preserves_false_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"scalar": 2, "null": null, "enabled": false, "count": 0}', encoding="utf-8"
    )
    loader = config_loader.ConfigLoader(path)
    assert loader.get("scalar.child", "fallback") == "fallback"
    assert loader.get("null", "fallback") == "fallback"
    assert loader.get("enabled", True) is False
    assert loader.get("count", 10) == 0


def test_set_before_load_preserves_existing_file_values_and_creates_nested_sections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"existing": 1}', encoding="utf-8")
    loader = config_loader.ConfigLoader(path)
    loader.set("display.colors.primary", "blue")
    assert loader.get("existing") == 1
    assert loader.get("display.colors.primary") == "blue"
    assert loader.save() is True
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "existing": 1,
        "display": {"colors": {"primary": "blue"}},
    }


def test_save_without_loading_does_not_create_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.json"
    assert config_loader.ConfigLoader(path).save() is False
    assert not path.exists()
    assert "No configuration loaded" in caplog.text


def test_save_failure_preserves_destination_directory(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.mkdir()
    loader = config_loader.ConfigLoader(path, {"fallback": True})
    assert loader.load() == {"fallback": True}
    assert loader.save() is False
    assert path.is_dir()
    assert not (tmp_path / ".config.json.tmp").exists()


@pytest.mark.parametrize(
    "constructor", [config_loader.ConfigLoader, config_loader.load_config]
)
def test_absent_config_path_is_rejected(constructor: Any) -> None:
    missing: Any = None
    with pytest.raises(ValueError, match="config_path must be provided"):
        constructor(missing)


@pytest.mark.parametrize("operation", ["load", "get", "set"])
def test_absent_method_input_is_rejected(operation: str, tmp_path: Path) -> None:
    loader = config_loader.ConfigLoader(tmp_path / "config.json")
    missing: Any = None
    with pytest.raises(ValueError, match="must be provided"):
        if operation == "load":
            loader.load(reload=missing)
        elif operation == "get":
            loader.get(missing)
        else:
            loader.set(missing, "value")


@pytest.mark.parametrize("appdata", [None, "", "configured"])
def test_windows_uses_appdata_or_home_and_ignores_xdg(
    appdata: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_loader, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    if appdata is None:
        monkeypatch.delenv("APPDATA", raising=False)
    else:
        monkeypatch.setenv("APPDATA", str(tmp_path / "roaming") if appdata else "")
    expected_base = tmp_path / "roaming" if appdata else tmp_path / "home" / ".config"
    expected = expected_base / "fixture-app"
    assert config_loader.get_xdg_config_dir("fixture-app") == expected
    assert not expected.exists()
