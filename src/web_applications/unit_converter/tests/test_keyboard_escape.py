"""Browser regression tests for unit-converter keyboard behavior."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect, sync_playwright

APP_DIRECTORY = Path(__file__).resolve().parents[1] / "unit-converter-app"
ESCAPE_CLEAR_DELAY_MS = 400


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    """Launch Playwright's bundled Chromium for real keyboard and DOM interaction."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def converter_page(browser: Browser) -> Iterator[Page]:
    """Open the real unit-converter page in an isolated browser context."""
    context = browser.new_context()
    page = context.new_page()
    page.goto(APP_DIRECTORY.joinpath("index.html").as_uri())
    page.evaluate(
        """() => document.addEventListener('keydown', event => {
          if (event.key === 'Escape') {
            window.escapeDefaultPrevented = event.defaultPrevented;
          }
        })"""
    )
    yield page
    context.close()


@pytest.mark.parametrize("active_input", ["fromValue", "toValue"])
def test_escape_clears_both_values_and_allows_a_later_conversion(
    converter_page: Page, active_input: str
) -> None:
    """Escape clears either focused value, survives debounce, then converts again."""
    from_value = converter_page.locator("#fromValue")
    to_value = converter_page.locator("#toValue")
    clear_button = converter_page.locator("#clearInput")
    converter_page.locator("#fromUnit").select_option("m")
    converter_page.locator("#toUnit").select_option("cm")

    from_value.fill("2")
    to_value.fill("200")
    converter_page.wait_for_timeout(ESCAPE_CLEAR_DELAY_MS)
    expect(clear_button).to_be_visible()

    from_value.fill("4")
    to_value.fill("400")
    converter_page.locator(f"#{active_input}").focus()
    converter_page.keyboard.press("Escape")

    expect(from_value).to_have_value("")
    expect(to_value).to_have_value("")
    expect(from_value).to_be_focused()
    expect(clear_button).to_have_class("clear-input-button hidden")
    assert converter_page.evaluate("window.escapeDefaultPrevented") is True

    converter_page.wait_for_timeout(ESCAPE_CLEAR_DELAY_MS)
    expect(from_value).to_have_value("")
    expect(to_value).to_have_value("")

    from_value.fill("3")
    expect(to_value).to_have_value("300")


def test_escape_elsewhere_preserves_values_and_modal_escape_still_closes(
    converter_page: Page,
) -> None:
    """Escape outside value fields is untouched, while Escape still closes the modal."""
    from_value = converter_page.locator("#fromValue")
    to_value = converter_page.locator("#toValue")
    from_value.fill("2")
    to_value.fill("200")
    converter_page.wait_for_timeout(ESCAPE_CLEAR_DELAY_MS)

    converter_page.locator("#category").focus()
    converter_page.keyboard.press("Escape")

    assert converter_page.evaluate("window.escapeDefaultPrevented") is False
    expect(from_value).not_to_have_value("")
    expect(to_value).not_to_have_value("")

    modal = converter_page.locator("#customUnitsModal")
    converter_page.locator("#customUnitsButton").click()
    expect(modal).to_be_visible()
    converter_page.keyboard.press("Escape")
    expect(modal).to_have_class("modal hidden")
