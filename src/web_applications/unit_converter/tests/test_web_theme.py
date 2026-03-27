"""Tests for the web theme bridge module."""

from __future__ import annotations

from typing import Any

import pytest

from web_applications.unit_converter.web_theme import (
    all_themes_as_css,
    get_default_theme_name,
    get_theme_by_name,
    get_theme_names,
    get_themes_for_api,
    load_themes,
    theme_to_css_vars,
)


class TestLoadThemes:
    def test_returns_dict(self) -> None:
        data = load_themes()
        assert isinstance(data, dict)

    def test_has_themes_key(self) -> None:
        data = load_themes()
        assert "themes" in data
        assert len(data["themes"]) > 0


class TestGetThemeNames:
    def test_returns_list(self) -> None:
        names = get_theme_names()
        assert isinstance(names, list)

    def test_includes_dark(self) -> None:
        names = get_theme_names()
        assert "Dark" in names

    def test_includes_light(self) -> None:
        names = get_theme_names()
        assert "Light" in names


class TestGetThemeByName:
    def test_dark_theme(self) -> None:
        theme = get_theme_by_name("Dark")
        assert theme is not None
        assert theme["isDark"] is True
        assert "colors" in theme

    def test_light_theme(self) -> None:
        theme = get_theme_by_name("Light")
        assert theme is not None
        assert theme["isDark"] is False

    def test_unknown_theme(self) -> None:
        theme = get_theme_by_name("NonexistentTheme12345")
        assert theme is None


class TestThemeToCssVars:
    def test_generates_css(self) -> None:
        theme = get_theme_by_name("Dark")
        assert theme is not None
        css = theme_to_css_vars(theme)
        assert "--bg:" in css
        assert "--text-primary:" in css
        assert "--accent:" in css

    def test_dark_theme_has_dark_color_scheme(self) -> None:
        theme = get_theme_by_name("Dark")
        assert theme is not None
        css = theme_to_css_vars(theme)
        assert "color-scheme: dark" in css

    def test_light_theme_has_light_color_scheme(self) -> None:
        theme = get_theme_by_name("Light")
        assert theme is not None
        css = theme_to_css_vars(theme)
        assert "color-scheme: light" in css

    def test_includes_semantic_colors(self) -> None:
        theme = get_theme_by_name("Dark")
        assert theme is not None
        css = theme_to_css_vars(theme)
        assert "--success:" in css
        assert "--error:" in css

    def test_maps_all_base_keys(self) -> None:
        theme = get_theme_by_name("Dark")
        assert theme is not None
        css = theme_to_css_vars(theme)
        expected_vars = [
            "--bg:",
            "--bg-card:",
            "--border:",
            "--text-primary:",
            "--text-secondary:",
            "--text-muted:",
            "--border-focus:",
            "--bg-input:",
            "--accent:",
            "--bg-elevated:",
            "--accent-hover:",
        ]
        for var in expected_vars:
            assert var in css, f"Missing CSS variable: {var}"


class TestAllThemesAsCss:
    def test_generates_css_string(self) -> None:
        css = all_themes_as_css()
        assert isinstance(css, str)
        assert len(css) > 100

    def test_has_root_block(self) -> None:
        css = all_themes_as_css()
        assert ":root {" in css

    def test_has_data_theme_selectors(self) -> None:
        css = all_themes_as_css()
        assert '[data-theme="Dark"]' in css
        assert '[data-theme="Light"]' in css
        assert '[data-theme="Dracula"]' in css


class TestGetThemesForApi:
    def test_returns_list(self) -> None:
        themes = get_themes_for_api()
        assert isinstance(themes, list)
        assert len(themes) > 0

    def test_each_theme_has_required_fields(self) -> None:
        themes = get_themes_for_api()
        for theme in themes:
            assert "id" in theme
            assert "name" in theme
            assert "isDark" in theme

    def test_includes_dark_theme(self) -> None:
        themes = get_themes_for_api()
        names = [t["name"] for t in themes]
        assert "Dark" in names


class TestGetDefaultThemeName:
    def test_returns_dark(self) -> None:
        assert get_default_theme_name() == "Dark"


class TestThemeApiEndpoints:
    """Test the Flask theme API endpoints."""

    @pytest.fixture
    def client(self) -> Any:
        from web_applications.unit_converter.webapp import create_app

        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_theme_css_endpoint(self, client) -> None:
        response = client.get("/api/theme.css")
        assert response.status_code == 200
        assert response.content_type == "text/css; charset=utf-8"
        css = response.data.decode()
        assert ":root {" in css
        assert '[data-theme="Dark"]' in css

    def test_themes_api_endpoint(self, client) -> None:
        response = client.get("/api/themes")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        names = [t["name"] for t in data]
        assert "Dark" in names
        assert "Light" in names

    def test_index_includes_theme_selector(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        html = response.data.decode()
        assert "themeSelect" in html
        assert "data-theme" in html

    def test_index_includes_theme_css_link(self, client) -> None:
        response = client.get("/")
        html = response.data.decode()
        assert "/api/theme.css" in html
