"""Widget-level tests for ChatQuickBar theme integration.

These tests verify that the quick-bar widget correctly applies theme colors
to each sub-widget and that ``apply_theme()`` propagates updates.  All tests
in this file require PyQt6.QtWidgets (display server) and pytest-qt; the
entire file is skipped when either is unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "PyQt6.QtWebSockets",
    reason="PyQt6.QtWebSockets DLL load failed",
    exc_type=ImportError,
)
pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6.QtWidgets requires display server")
pytest.importorskip("pytestqt", reason="pytest-qt required for widget tests")


class _FixedThemeProvider:
    """Test double that returns a user-supplied color dict."""

    def __init__(self, colors: dict[str, str]) -> None:
        self._colors = colors

    def get_colors(self) -> dict[str, str]:
        return dict(self._colors)


def _track_widget(qtbot, widget) -> None:
    try:
        qtbot.addWidget(widget)
    except TypeError:
        from PyQt6.QtCore import Qt

        widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)


class TestChatQuickBarThemeIntegration:
    """Widget-level tests for ``ChatQuickBar`` theme integration."""

    def test_construction_with_fake_provider(self, qtbot) -> None:
        from chat.quick_bar import ChatQuickBar

        provider = _FixedThemeProvider({"accent": "#deadbe"})
        bar = ChatQuickBar(theme_provider=provider)
        _track_widget(qtbot, bar)
        assert bar._theme is provider

    def test_stylesheet_contains_theme_accent(self, qtbot) -> None:
        from chat.quick_bar import ChatQuickBar

        provider = _FixedThemeProvider({"accent": "#aabbcc"})
        bar = ChatQuickBar(theme_provider=provider)
        _track_widget(qtbot, bar)
        send_style = bar._send_btn.styleSheet()
        assert "#aabbcc" in send_style

    def test_stylesheet_contains_theme_background(self, qtbot) -> None:
        from chat.quick_bar import ChatQuickBar

        provider = _FixedThemeProvider({"background_primary": "#111111"})
        bar = ChatQuickBar(theme_provider=provider)
        _track_widget(qtbot, bar)
        frame_style = bar.styleSheet()
        assert "#111111" in frame_style

    def test_construction_with_default_provider(self, qtbot) -> None:
        """Default construction (no provider) should still produce a stylesheet."""
        from chat.quick_bar import ChatQuickBar

        bar = ChatQuickBar()
        _track_widget(qtbot, bar)
        assert "background-color" in bar.styleSheet()

    def test_apply_theme_updates_stylesheet(self, qtbot) -> None:
        from chat.quick_bar import ChatQuickBar

        class _MutableProvider:
            def __init__(self) -> None:
                self._accent = "#ff0000"

            def get_colors(self) -> dict[str, str]:
                return {"accent": self._accent}

        provider = _MutableProvider()
        bar = ChatQuickBar(theme_provider=provider)
        _track_widget(qtbot, bar)

        assert "#ff0000" in bar._send_btn.styleSheet()

        provider._accent = "#00ff00"
        bar.apply_theme()
        assert "#00ff00" in bar._send_btn.styleSheet()
        assert "#ff0000" not in bar._send_btn.styleSheet()

    def test_input_field_uses_theme_colors(self, qtbot) -> None:
        from chat.quick_bar import ChatQuickBar

        provider = _FixedThemeProvider(
            {
                "input_background": "#222222",
                "text_primary": "#eeeeee",
                "border": "#444444",
            }
        )
        bar = ChatQuickBar(theme_provider=provider)
        _track_widget(qtbot, bar)
        input_style = bar._input.styleSheet()
        assert "#222222" in input_style
        assert "#eeeeee" in input_style
        assert "#444444" in input_style
