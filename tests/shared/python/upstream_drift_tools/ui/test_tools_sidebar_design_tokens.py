"""Tests for the Sidekick sidebar design-token contract."""

from __future__ import annotations

import pytest
from upstream_drift_tools.ui.tools_sidebar import (
    SIDEKICK_SIDEBAR_OBJECT_NAME,
    SIDEKICK_TABS_OBJECT_NAME,
    SIDEKICK_TOKEN_NAMES,
    SIDEKICK_TOOLBAR_OBJECT_NAME,
    SidebarState,
    SidekickDesignTokens,
    SidekickTerminalTheme,
    SidekickThemeMode,
    SidekickThemeSettings,
    resolve_sidekick_theme,
    sidekick_qss,
)


def test_sidekick_design_tokens_export_css_and_qss_mappings() -> None:
    tokens = SidekickDesignTokens()

    css_variables = tokens.css_variables()
    qss_variables = tokens.qss_variables()

    assert SIDEKICK_TOKEN_NAMES == tuple(tokens.values)
    assert css_variables["--sidekick-color-background"] == "#f7f9fc"
    assert css_variables["--sidekick-color-accent"] == tokens["color.accent"]
    assert css_variables["--sidekick-color-focus"] == "#3b82f6"
    assert css_variables["--sidekick-font-family"] == "Arial"
    assert css_variables["--sidekick-shadow-panel"].startswith("0 12px")
    assert qss_variables["sidekick-radius-panel"] == "8px"
    assert qss_variables["sidekick-space-2"] == "8px"
    assert qss_variables["sidekick-size-control-height"] == "28px"
    assert qss_variables["sidekick-font-size"] == "12px"


def test_sidekick_qss_uses_stable_selectors_and_token_overrides() -> None:
    tokens = SidekickDesignTokens({"color.background": "#f8fafc"})

    qss = sidekick_qss(tokens)

    assert f"QWidget#{SIDEKICK_SIDEBAR_OBJECT_NAME}" in qss
    assert f"QToolBar#{SIDEKICK_TOOLBAR_OBJECT_NAME}" in qss
    assert f"QTabWidget#{SIDEKICK_TABS_OBJECT_NAME}::pane" in qss
    assert "QToolButton:focus" in qss
    assert "#f8fafc" in qss


def test_sidekick_design_tokens_accept_host_sidekick_names() -> None:
    tokens = SidekickDesignTokens.from_sidekick_tokens(
        {
            "sidekick.color.canvas": "#101820",
            "sidekick.color.surface.elevated": "#ffffff",
            "sidekick.color.error": "#b42318",
            "sidekick.radius.chat": "7px",
            "sidekick.control.height": "32px",
        }
    )

    assert tokens["color.background"] == "#101820"
    assert tokens["color.surface.raised"] == "#ffffff"
    assert tokens["color.danger"] == "#b42318"
    assert tokens["radius.panel"] == "7px"
    assert tokens["size.control.height"] == "32px"


def test_sidekick_design_tokens_can_derive_from_shared_schema() -> None:
    tokens = SidekickDesignTokens.from_shared_theme("dark")

    assert tokens["color.background"] == "#1a1d23"
    assert tokens["color.surface"] == "#24272e"
    assert tokens["color.accent"] == "#4a7ba7"
    assert tokens["radius.control"] == "6px"
    assert tokens["radius.panel"] == "8px"


def test_sidekick_design_tokens_reject_blank_required_values() -> None:
    with pytest.raises(ValueError, match="color.background"):
        SidekickDesignTokens({"color.background": ""})


def test_sidekick_terminal_theme_inherits_resolved_tokens() -> None:
    tokens = SidekickDesignTokens(
        {
            "color.text": "#111111",
            "color.surface": "#222222",
            "color.accent": "#333333",
            "color.selection": "#444444",
        },
    )

    terminal_theme = SidekickTerminalTheme.inherited(tokens)

    assert terminal_theme.mode == "inherit"
    assert terminal_theme["foreground"] == "#111111"
    assert terminal_theme["background"] == "#222222"
    assert terminal_theme["cursor"] == "#333333"
    assert terminal_theme["selection"] == "#444444"
    assert "QWidget#SidekickTerminalTab QPlainTextEdit" in terminal_theme.qss(
        "SidekickTerminalTab",
    )


def test_sidekick_terminal_theme_accepts_custom_palette() -> None:
    terminal_theme = SidekickTerminalTheme.custom(
        foreground="#f8fafc",
        background="#020617",
        cursor="#38bdf8",
        selection="#1e293b",
        ansi={"red": "#ef4444"},
    )

    assert terminal_theme.mode == "custom"
    assert terminal_theme["foreground"] == "#f8fafc"
    assert terminal_theme["ansi.red"] == "#ef4444"
    assert "background: #020617" in terminal_theme.qss("SidekickTerminalTab")


def test_sidekick_terminal_theme_rejects_invalid_colors() -> None:
    with pytest.raises(ValueError, match="foreground"):
        SidekickTerminalTheme.custom(
            foreground="not-a-color",
            background="#020617",
        )


def test_sidekick_theme_settings_default_to_inherited_parent_tokens() -> None:
    parent = SidekickDesignTokens({"color.background": "#101820"})

    resolved = resolve_sidekick_theme(parent_tokens=parent)

    assert SidekickThemeSettings().mode is SidekickThemeMode.INHERIT_PARENT
    assert resolved["color.background"] == "#101820"


def test_sidekick_custom_theme_overrides_color_and_font_tokens() -> None:
    parent = SidekickDesignTokens({"color.background": "#101820"})
    settings = SidekickThemeSettings.custom(
        colors={"color.background": "#f8fafc", "sidekick.color.accent": "#0ea5e9"},
        font_family="Inter",
        font_size_px=14,
    )

    resolved = resolve_sidekick_theme(parent_tokens=parent, settings=settings)

    assert resolved["color.background"] == "#f8fafc"
    assert resolved["color.accent"] == "#0ea5e9"
    assert resolved["font.family"] == "Inter"
    assert resolved["font.size"] == "14px"


def test_sidekick_theme_settings_round_trip_through_sidebar_state() -> None:
    state = SidebarState(
        theme_settings=SidekickThemeSettings.custom(
            colors={"color.surface": "#111827"},
            font_family="Source Sans 3",
            font_size_px=13,
        ).to_dict()
    )

    restored = SidebarState.from_dict(state.to_dict())

    assert restored.theme_settings["mode"] == "custom"
    assert restored.theme_settings["custom"]["colors"] == {"color.surface": "#111827"}
    assert restored.theme_settings["custom"]["font"] == {
        "family": "Source Sans 3",
        "size_px": 13,
    }


def test_sidekick_theme_settings_reject_invalid_payloads() -> None:
    with pytest.raises(ValueError, match="theme mode"):
        SidekickThemeSettings.from_dict({"mode": "parent"})
    with pytest.raises(ValueError, match="color.background"):
        SidekickThemeSettings.custom(colors={"color.background": "blue"})
    with pytest.raises(ValueError, match="font size"):
        SidekickThemeSettings.custom(font_size_px=72)
    with pytest.raises(ValueError, match="font family"):
        SidekickThemeSettings.custom(font_family="x" * 65)
