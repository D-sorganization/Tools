"""Tests for the shared FastAPI theme router."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.shared.python.theme.api import (
    SaveCustomThemeRequest,
    SetActiveThemeRequest,
    ThemeColors,
    _register_active_and_list_endpoints,
    _register_custom_endpoints,
    create_theme_router,
)


class FakeThemeManager:
    def __init__(self) -> None:
        self.builtin = {
            "dark": {"bg": "#000000", "text": "#ffffff"},
            "empty": {},
        }
        self.custom = {
            "ocean": {"bg": "#003344", "text": "#dff8ff"},
            "blank": {},
        }
        self.current = "dark"
        self.saved: tuple[str, dict[str, str], bool] | None = None
        self.changed_to: str | None = None

    def get_builtin_themes(self) -> list[str]:
        return list(self.builtin)

    def get_custom_theme_names(self) -> list[str]:
        return list(self.custom)

    def get_theme_colors(self, name: str) -> dict[str, str]:
        return self.builtin.get(name) or self.custom.get(name) or {}

    def get_current_theme_name(self) -> str:
        return self.current

    def get_current_colors(self) -> dict[str, str]:
        return self.get_theme_colors(self.current)

    def get_available_themes(self) -> list[str]:
        return [*self.builtin, *self.custom]

    def change_theme(self, name: str) -> None:
        self.current = name
        self.changed_to = name

    def save_custom_theme(self, name: str, colors: dict[str, str], apply: bool) -> str:
        if name == "bad":
            raise ValueError("invalid theme")
        self.saved = (name, colors, apply)
        self.custom[name] = colors
        if apply:
            self.current = name
        return name

    def delete_custom_theme(self, name: str) -> bool:
        if name not in self.custom:
            return False
        del self.custom[name]
        return True


def _client(manager: FakeThemeManager) -> TestClient:
    app = FastAPI()
    app.include_router(create_theme_router(manager), prefix="/themes")
    return TestClient(app)


def test_theme_models_preserve_color_and_request_contracts() -> None:
    colors = ThemeColors(
        bg="#000000",
        group_bg="#111111",
        border="#222222",
        text="#ffffff",
        text_secondary="#dddddd",
        label="#bbbbbb",
        focus="#3b82f6",
        input_bg="#101010",
        accent="#10b981",
        title_bg="#050505",
        title_border="#333333",
        table_header="#181818",
        table_alt="#121212",
        button_hover="#242424",
    )
    set_active = SetActiveThemeRequest(name="dark")
    save_custom = SaveCustomThemeRequest(
        name="custom", colors={"bg": "#123456"}, apply=True
    )

    assert colors.accent == "#10b981"
    assert set_active.name == "dark"
    assert save_custom.apply is True


def test_lists_builtin_custom_and_all_themes_filtering_empty_definitions() -> None:
    client = _client(FakeThemeManager())

    builtin = client.get("/themes/builtin").json()["themes"]
    custom = client.get("/themes/custom").json()["themes"]
    all_themes = client.get("/themes/").json()["themes"]

    assert builtin == {
        "dark": {
            "name": "dark",
            "is_builtin": True,
            "colors": {"bg": "#000000", "text": "#ffffff"},
        }
    }
    assert custom["ocean"]["is_builtin"] is False
    assert "empty" not in all_themes
    assert "blank" not in all_themes


def test_active_theme_get_and_set_success() -> None:
    manager = FakeThemeManager()
    client = _client(manager)

    active = client.get("/themes/active").json()
    response = client.put("/themes/active", json={"name": "ocean"})

    assert active == {
        "name": "dark",
        "is_builtin": True,
        "colors": {"bg": "#000000", "text": "#ffffff"},
    }
    assert response.status_code == 200
    assert response.json()["theme_name"] == "ocean"
    assert manager.changed_to == "ocean"


def test_set_active_theme_rejects_unknown_theme_with_available_names() -> None:
    client = _client(FakeThemeManager())

    response = client.put("/themes/active", json={"name": "missing"})

    assert response.status_code == 404
    assert "Available: dark, empty, ocean, blank" in response.json()["detail"]


def test_save_custom_theme_success_and_validation_failure() -> None:
    manager = FakeThemeManager()
    client = _client(manager)

    response = client.post(
        "/themes/custom",
        json={"name": "forest", "colors": {"bg": "#123456"}, "apply": True},
    )
    failed = client.post(
        "/themes/custom",
        json={"name": "bad", "colors": {"bg": "#123456"}, "apply": False},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "Theme 'forest' saved successfully",
        "theme_name": "forest",
    }
    assert manager.saved == ("forest", {"bg": "#123456"}, True)
    assert failed.status_code == 400
    assert failed.json()["detail"] == "invalid theme"


def test_delete_custom_theme_success_and_not_found() -> None:
    client = _client(FakeThemeManager())

    deleted = client.delete("/themes/custom/ocean")
    missing = client.delete("/themes/custom/missing")

    assert deleted.status_code == 200
    assert deleted.json()["message"] == "Theme 'ocean' deleted"
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Custom theme 'missing' not found"


def test_endpoint_registration_rejects_missing_router() -> None:
    manager = FakeThemeManager()

    with pytest.raises(ValueError, match="router must be provided"):
        _register_custom_endpoints(None, manager)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="router must be provided"):
        _register_active_and_list_endpoints(None, manager)  # type: ignore[arg-type]
