"""Tests for the Sidekick sidebar design-token contract."""

from __future__ import annotations

import pytest
from upstream_drift_tools.ui.tools_sidebar import (
    SIDEKICK_SIDEBAR_OBJECT_NAME,
    SIDEKICK_TABS_OBJECT_NAME,
    SIDEKICK_TOKEN_NAMES,
    SIDEKICK_TOOLBAR_OBJECT_NAME,
    SidekickDesignTokens,
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
    assert css_variables["--sidekick-shadow-panel"].startswith("0 12px")
    assert qss_variables["sidekick-radius-panel"] == "8px"
    assert qss_variables["sidekick-space-2"] == "8px"
    assert qss_variables["sidekick-size-control-height"] == "28px"


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
