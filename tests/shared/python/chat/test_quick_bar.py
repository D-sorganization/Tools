"""Tests for ChatQuickBar theme helpers (pure Python, no Qt dependency).

These tests import only from ``chat._quick_bar_theme`` which has zero Qt
dependency and can therefore run in any environment including headless CI
and Windows machines without a working PyQt6.QtWebSockets DLL.
"""

from __future__ import annotations

import pytest
from chat._quick_bar_theme import (
    ThemeProviderProtocol,
    _FallbackThemeProvider,
    _resolve_colors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedThemeProvider:
    """Test double that returns a user-supplied color dict."""

    def __init__(self, colors: dict[str, str]) -> None:
        self._colors = colors

    def get_colors(self) -> dict[str, str]:
        return dict(self._colors)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestThemeProviderProtocol:
    """Ensure built-in providers satisfy the protocol at runtime."""

    def test_fallback_provider_satisfies_protocol(self) -> None:
        provider = _FallbackThemeProvider()
        assert isinstance(provider, ThemeProviderProtocol)

    def test_fixed_provider_satisfies_protocol(self) -> None:
        provider = _FixedThemeProvider({})
        assert isinstance(provider, ThemeProviderProtocol)

    def test_fallback_returns_all_expected_keys(self) -> None:
        provider = _FallbackThemeProvider()
        colors = provider.get_colors()
        expected_keys = {
            "background_primary",
            "background_secondary",
            "border",
            "text_primary",
            "text_secondary",
            "accent",
            "accent_hover",
            "focus",
            "input_background",
            "button_background",
            "button_hover",
            "disabled_background",
            "disabled_foreground",
            "muted",
        }
        assert expected_keys.issubset(set(colors.keys()))


# ---------------------------------------------------------------------------
# Color resolution
# ---------------------------------------------------------------------------


class TestResolveColors:
    """Unit tests for ``_resolve_colors()``."""

    def test_uses_theme_token_when_present(self) -> None:
        provider = _FixedThemeProvider({"accent": "#aabbcc"})
        c = _resolve_colors(provider)
        assert c["accent"] == "#aabbcc"

    def test_uses_alias_token(self) -> None:
        """``text_primary`` should map to the ``text`` canonical key."""
        provider = _FixedThemeProvider({"text_primary": "#ffffff"})
        c = _resolve_colors(provider)
        assert c["text"] == "#ffffff"

    def test_falls_back_to_hex_when_token_absent(self) -> None:
        provider = _FixedThemeProvider({})
        c = _resolve_colors(provider)
        for key, value in c.items():
            assert value.startswith("#"), f"Key {key!r} has non-hex fallback: {value!r}"

    def test_all_canonical_keys_present(self) -> None:
        provider = _FixedThemeProvider({})
        c = _resolve_colors(provider)
        required = {
            "bg",
            "group_bg",
            "border",
            "text",
            "text_secondary",
            "accent",
            "accent_hover",
            "focus",
            "input_bg",
            "button_bg",
            "button_hover",
            "disabled_bg",
            "disabled_fg",
            "muted",
        }
        assert required.issubset(set(c.keys()))

    def test_fallback_provider_produces_coherent_palette(self) -> None:
        c = _resolve_colors(_FallbackThemeProvider())
        for key, value in c.items():
            assert len(value) in (4, 7, 9), (
                f"Key {key!r} resolved to non-standard color: {value!r}"
            )
            assert value.startswith("#"), (
                f"Key {key!r} resolved to non-hex color: {value!r}"
            )

    def test_partial_theme_uses_fallback_for_missing_keys(self) -> None:
        """A partial palette should not break resolution for missing tokens."""
        provider = _FixedThemeProvider({"accent": "#123456"})
        c = _resolve_colors(provider)
        assert c["accent"] == "#123456"
        assert c["bg"].startswith("#")
        assert c["border"].startswith("#")

    @pytest.mark.parametrize(
        ("token", "canonical"),
        [
            ("background_primary", "bg"),
            ("background_secondary", "group_bg"),
            ("text_primary", "text"),
            ("input_background", "input_bg"),
            ("button_background", "button_bg"),
            ("disabled_background", "disabled_bg"),
            ("disabled_foreground", "disabled_fg"),
        ],
    )
    def test_token_aliases(self, token: str, canonical: str) -> None:
        """Theme manager tokens map to the correct canonical keys."""
        provider = _FixedThemeProvider({token: "#abcdef"})
        c = _resolve_colors(provider)
        assert c[canonical] == "#abcdef"
