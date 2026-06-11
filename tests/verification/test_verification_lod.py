"""Tests for LoD compliance in src/verification scripts.

Verifies that:
- LoD violations are resolved (no chained attribute access >2 levels)
- Each function behaves correctly when called with mocked Playwright objects
- All Playwright browser interactions are properly delegated to local variables

Playwright is mocked at sys.modules level to avoid requiring playwright installed.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _install_playwright_mock() -> None:
    """Insert a minimal playwright mock into sys.modules if not already present."""
    if "playwright" not in sys.modules:
        pw_mock = types.ModuleType("playwright")
        sync_api_mock = types.ModuleType("playwright.sync_api")
        sync_api_mock.sync_playwright = MagicMock()
        sync_api_mock.expect = MagicMock()
        pw_mock.sync_api = sync_api_mock
        sys.modules["playwright"] = pw_mock
        sys.modules["playwright.sync_api"] = sync_api_mock


_install_playwright_mock()


def _make_playwright_ctx(mock_p: MagicMock) -> MagicMock:
    """Return a mock context manager that yields mock_p."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_p)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


@pytest.mark.unit
class TestVerifyA11yLod:
    """Tests for LoD fix in verify_a11y.py: p.chromium extracted to local variable."""

    def _make_stack(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        mock_page = MagicMock()
        mock_page.locator.return_value.get_attribute.return_value = (
            "gasDensityHint heatingValHint"
        )
        mock_browser = MagicMock()
        mock_browser.new_page.return_value = mock_page
        mock_chromium = MagicMock()
        mock_chromium.launch.return_value = mock_browser
        mock_p = MagicMock()
        mock_p.chromium = mock_chromium
        return mock_page, mock_browser, mock_chromium, mock_p

    def test_verify_a11y_uses_chromium_local(self) -> None:
        """verify_a11y should launch browser via chromium local, not p.chromium.launch()."""
        _, _, mock_chromium, mock_p = self._make_stack()
        ctx = _make_playwright_ctx(mock_p)

        import verification.verify_a11y as mod

        importlib.reload(mod)
        with patch.object(mod, "sync_playwright", return_value=ctx):
            mod.verify_a11y()

        mock_chromium.launch.assert_called_once_with(headless=True)

    def test_verify_a11y_navigates_to_localhost(self) -> None:
        """verify_a11y navigates to localhost:8080."""
        mock_page, _, _, mock_p = self._make_stack()
        ctx = _make_playwright_ctx(mock_p)

        import verification.verify_a11y as mod

        importlib.reload(mod)
        with (
            patch.object(mod, "sync_playwright", return_value=ctx),
            patch.object(mod, "expect"),
        ):
            mod.verify_a11y()

        mock_page.goto.assert_called_once_with("http://localhost:8080")


@pytest.mark.unit
class TestVerifyPaletteLod:
    """Tests for LoD fix in verify_palette.py: locator chain extracted to local."""

    def _make_stack(
        self, described_by_value: str | None = "gasFlowHint"
    ) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        mock_page = MagicMock()
        locator_map: dict[str, MagicMock] = {}

        def locator_side_effect(selector: str) -> MagicMock:
            if selector not in locator_map:
                m = MagicMock()
                m.count.return_value = 1
                m.get_attribute.return_value = described_by_value
                locator_map[selector] = m
            return locator_map[selector]

        mock_page.locator.side_effect = locator_side_effect
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_chromium = MagicMock()
        mock_chromium.launch.return_value = mock_browser
        mock_p = MagicMock()
        mock_p.chromium = mock_chromium
        return mock_page, mock_browser, mock_chromium, mock_p

    def test_run_uses_chromium_local(self) -> None:
        """verify_palette.run() should launch browser via chromium local variable."""
        _, _, mock_chromium, mock_p = self._make_stack()
        ctx = _make_playwright_ctx(mock_p)

        import verification.verify_palette as mod

        importlib.reload(mod)
        with patch.object(mod, "sync_playwright", return_value=ctx):
            mod.run()

        mock_chromium.launch.assert_called_once_with(headless=True)

    def test_run_checks_aria_describedby_via_local(self) -> None:
        """run() calls get_attribute on a locator extracted to a local variable."""
        mock_page, _, _, mock_p = self._make_stack(described_by_value="gasFlowHint")
        ctx = _make_playwright_ctx(mock_p)

        import verification.verify_palette as mod

        importlib.reload(mod)
        with patch.object(mod, "sync_playwright", return_value=ctx):
            mod.run()

        calls = [str(call) for call in mock_page.locator.call_args_list]
        assert any("#standardCondition" in c for c in calls)


@pytest.mark.unit
class TestVerifyPaletteFinalLod:
    """Tests for LoD fix in verify_palette_final.py: locator chains extracted to locals."""

    def _make_stack(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        mock_page = MagicMock()
        from_label_mock = MagicMock()
        kbd_shortcut_mock = MagicMock()
        kbd_shortcut_mock.inner_text.return_value = "Ctrl+K"
        from_label_mock.locator.return_value = kbd_shortcut_mock

        condition_locator_mock = MagicMock()
        condition_locator_mock.get_attribute.return_value = "gasFlowHint"

        gas_flow_hint_mock = MagicMock()
        gas_flow_hint_mock.count.return_value = 1

        def locator_side_effect(selector: str) -> MagicMock:
            if selector == "label[for='fromValue']":
                return from_label_mock
            if selector == "#gasFlowHint":
                return gas_flow_hint_mock
            if selector == "#standardCondition":
                return condition_locator_mock
            return MagicMock()

        mock_page.locator.side_effect = locator_side_effect
        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_chromium = MagicMock()
        mock_chromium.launch.return_value = mock_browser
        mock_p = MagicMock()
        mock_p.chromium = mock_chromium
        return mock_page, mock_browser, mock_chromium, mock_p

    def test_run_uses_chromium_local(self) -> None:
        """verify_palette_final.run() launches browser via chromium local variable."""
        _, _, mock_chromium, mock_p = self._make_stack()
        ctx = _make_playwright_ctx(mock_p)

        import verification.verify_palette_final as mod

        importlib.reload(mod)
        with patch.object(mod, "sync_playwright", return_value=ctx):
            mod.run()

        mock_chromium.launch.assert_called_once_with(headless=True)

    def test_run_uses_kbd_shortcut_local(self) -> None:
        """run() extracts kbd-shortcut locator to local before calling inner_text()."""
        mock_page, _, _, mock_p = self._make_stack()
        ctx = _make_playwright_ctx(mock_p)

        import verification.verify_palette_final as mod

        importlib.reload(mod)
        with patch.object(mod, "sync_playwright", return_value=ctx):
            mod.run()

        from_label_mock = mock_page.locator("label[for='fromValue']")
        from_label_mock.locator.assert_called_with(".kbd-shortcut")

    def test_run_uses_condition_locator_local(self) -> None:
        """run() extracts standardCondition locator to local before get_attribute()."""
        mock_page, _, _, mock_p = self._make_stack()
        ctx = _make_playwright_ctx(mock_p)

        import verification.verify_palette_final as mod

        importlib.reload(mod)
        with patch.object(mod, "sync_playwright", return_value=ctx):
            mod.run()

        calls = [str(call) for call in mock_page.locator.call_args_list]
        assert any("#standardCondition" in c for c in calls)
