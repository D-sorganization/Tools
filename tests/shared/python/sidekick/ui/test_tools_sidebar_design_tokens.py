"""Tests for Sidekick design token system.

DbC: Each test states preconditions and postconditions.
LOD: Tests use the public SidekickDesignTokens API only.
"""

from __future__ import annotations

import pytest


class TestSidekickDesignTokens:
    """Verify SidekickDesignTokens construction, lookup, and CSS emission."""

    def test_default_tokens_importable(self) -> None:
        """Precondition: sidekick package is installed.
        Postcondition: SIDEKICK_DESIGN_TOKENS is importable and non-None."""
        from sidekick.ui.tools_sidebar.design_tokens import (
            SIDEKICK_DESIGN_TOKENS,
            SidekickDesignTokens,
        )

        assert SIDEKICK_DESIGN_TOKENS is not None
        assert isinstance(SIDEKICK_DESIGN_TOKENS, SidekickDesignTokens)

    def test_token_names_tuple_is_non_empty(self) -> None:
        """Precondition: module is imported.
        Postcondition: SIDEKICK_TOKEN_NAMES is a non-empty tuple."""
        from sidekick.ui.tools_sidebar.design_tokens import SIDEKICK_TOKEN_NAMES

        assert isinstance(SIDEKICK_TOKEN_NAMES, tuple)
        assert len(SIDEKICK_TOKEN_NAMES) > 0

    def test_all_default_token_names_have_values(self) -> None:
        """Precondition: SIDEKICK_DESIGN_TOKENS was constructed from defaults.
        Postcondition: every token name in SIDEKICK_TOKEN_NAMES returns
        a non-empty value."""
        from sidekick.ui.tools_sidebar.design_tokens import (
            SIDEKICK_DESIGN_TOKENS,
            SIDEKICK_TOKEN_NAMES,
        )

        for name in SIDEKICK_TOKEN_NAMES:
            value = SIDEKICK_DESIGN_TOKENS[name]
            assert value, f"Token {name!r} has empty/falsy value"

    def test_css_variables_all_prefixed_with_sidekick(self) -> None:
        """Precondition: SIDEKICK_DESIGN_TOKENS instance available.
        Postcondition: all css_variables() keys start with '--sidekick-'."""
        from sidekick.ui.tools_sidebar.design_tokens import SIDEKICK_DESIGN_TOKENS

        css_vars = SIDEKICK_DESIGN_TOKENS.css_variables()
        assert isinstance(css_vars, dict)
        assert len(css_vars) > 0
        for key in css_vars:
            assert key.startswith("--sidekick-"), f"Bad CSS var key: {key!r}"

    def test_qss_variables_prefixed_without_dashes(self) -> None:
        """Precondition: SIDEKICK_DESIGN_TOKENS instance available.
        Postcondition: qss_variables() keys start with 'sidekick-' (no double dash)."""
        from sidekick.ui.tools_sidebar.design_tokens import SIDEKICK_DESIGN_TOKENS

        qss_vars = SIDEKICK_DESIGN_TOKENS.qss_variables()
        assert isinstance(qss_vars, dict)
        assert len(qss_vars) > 0
        for key in qss_vars:
            assert key.startswith("sidekick-"), f"Bad QSS var key: {key!r}"
            assert not key.startswith("--"), f"QSS key has CSS double dash: {key!r}"

    def test_with_overrides_replaces_accent(self) -> None:
        """Precondition: SIDEKICK_DESIGN_TOKENS has a default 'color.accent'.
        Postcondition: with_overrides produces a new instance with the override
        applied while the original is unchanged."""
        from sidekick.ui.tools_sidebar.design_tokens import SIDEKICK_DESIGN_TOKENS

        original_accent = SIDEKICK_DESIGN_TOKENS["color.accent"]
        override_color = "#abcdef"

        custom = SIDEKICK_DESIGN_TOKENS.with_overrides(
            **{"color.accent": override_color}
        )

        assert custom["color.accent"] == override_color
        assert SIDEKICK_DESIGN_TOKENS["color.accent"] == original_accent

    def test_with_overrides_returns_sidekick_design_tokens(self) -> None:
        """Precondition: with_overrides called on an existing instance.
        Postcondition: returned value is also a SidekickDesignTokens instance."""
        from sidekick.ui.tools_sidebar.design_tokens import (
            SIDEKICK_DESIGN_TOKENS,
            SidekickDesignTokens,
        )

        result = SIDEKICK_DESIGN_TOKENS.with_overrides()
        assert isinstance(result, SidekickDesignTokens)

    def test_construct_from_sidekick_tokens_classmethod(self) -> None:
        """Precondition: valid sidekick.* alias names used.
        Postcondition: from_sidekick_tokens() accepts aliased names."""
        from sidekick.ui.tools_sidebar.design_tokens import SidekickDesignTokens

        tokens = SidekickDesignTokens.from_sidekick_tokens(
            {"sidekick.color.accent": "#123456"}
        )
        assert tokens["color.accent"] == "#123456"

    def test_object_names_are_string_constants(self) -> None:
        """Precondition: design_tokens module is imported.
        Postcondition: all SIDEKICK_*_OBJECT_NAME constants are non-empty strings."""
        import sidekick.ui.tools_sidebar.design_tokens as dt

        object_name_attrs = [
            attr
            for attr in dir(dt)
            if attr.startswith("SIDEKICK_") and attr.endswith("_OBJECT_NAME")
        ]
        assert len(object_name_attrs) > 0
        for attr in object_name_attrs:
            value = getattr(dt, attr)
            assert isinstance(value, str), f"{attr} is not a string"
            assert len(value) > 0, f"{attr} is empty"

    def test_sidekick_qss_returns_non_empty_string(self) -> None:
        """Precondition: sidekick_qss() called with no arguments.
        Postcondition: returns a non-empty QSS stylesheet string."""
        from sidekick.ui.tools_sidebar.design_tokens import sidekick_qss

        qss = sidekick_qss()
        assert isinstance(qss, str)
        assert len(qss) > 0

    def test_qss_defines_tab_hover_rule(self) -> None:
        """Precondition: sidekick_qss() emits the tab-bar stylesheet.
        Postcondition: a ``::tab`` hover rule exists so unselected tabs
        highlight on hover (regression: hover styling was missing while
        only ``::tab`` and ``::tab:selected`` were defined)."""
        from sidekick.ui.tools_sidebar.design_tokens import (
            SIDEKICK_TAB_BAR_OBJECT_NAME,
            sidekick_qss,
        )

        qss = sidekick_qss()
        hover_selector = f"QTabBar#{SIDEKICK_TAB_BAR_OBJECT_NAME}::tab"
        # The hover rule targets unselected tabs and must be present.
        assert f"{hover_selector}:!selected:hover" in qss or (
            f"{hover_selector}:hover" in qss
        ), "Sidekick QSS is missing a tab :hover rule"

    def test_terminal_theme_inherited_has_foreground_background(self) -> None:
        """Precondition: SidekickTerminalTheme.inherited() called.
        Postcondition: 'foreground' and 'background' keys are present."""
        from sidekick.ui.tools_sidebar.design_tokens import (
            SIDEKICK_DESIGN_TOKENS,
            SidekickTerminalTheme,
        )

        theme = SidekickTerminalTheme.inherited(SIDEKICK_DESIGN_TOKENS)
        assert "foreground" in theme.values
        assert "background" in theme.values

    def test_terminal_theme_mode_custom_validates_hex_colors(self) -> None:
        """Precondition: SidekickTerminalTheme.custom() called with valid hex colors.
        Postcondition: no exception raised, mode is 'custom'."""
        from sidekick.ui.tools_sidebar.design_tokens import SidekickTerminalTheme

        theme = SidekickTerminalTheme.custom(
            foreground="#ffffff",
            background="#000000",
        )
        assert theme.mode == "custom"
        assert theme.values["foreground"] == "#ffffff"

    def test_terminal_theme_invalid_mode_raises(self) -> None:
        """Precondition: SidekickTerminalTheme given invalid mode string.
        Postcondition: ValueError raised."""
        from sidekick.ui.tools_sidebar.design_tokens import SidekickTerminalTheme

        with pytest.raises(ValueError, match="inherit.*custom"):
            SidekickTerminalTheme(mode="invalid")
