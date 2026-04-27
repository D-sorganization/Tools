"""Final verification script for Unit Converter accessibility attributes and keyboard shortcuts."""

import logging
import os

from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)


def run() -> None:
    with sync_playwright() as p:
        chromium = p.chromium
        browser = chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Load the local HTML file
        cwd = os.getcwd()
        file_path = f"file://{cwd}/src/web_applications/unit_converter/unit-converter-app/index.html"
        logger.info(f"Loading: {file_path}")
        page.goto(file_path)

        # 1. Verify Keyboard Shortcut Hint
        logger.info("Verifying Keyboard Shortcut Hint...")
        from_label = page.locator("label[for='fromValue']")
        # Screenshot the label area
        from_label.screenshot(path="verification/shortcut_hint_final.png")
        logger.info("Screenshot saved to verification/shortcut_hint_final.png")

        # Verify the shortcut is Ctrl+K
        kbd_shortcut = from_label.locator(".kbd-shortcut")
        hint_text = kbd_shortcut.inner_text()
        logger.info(f"Shortcut hint text: {hint_text}")
        if "Ctrl+K" in hint_text:
            logger.info("SUCCESS: Shortcut hint is correct")
        else:
            logger.error("ERROR: Shortcut hint is incorrect")

        # 2. Verify Accessibility Attributes
        logger.info("Verifying Accessibility Attributes...")
        gas_flow_hint = page.locator("#gasFlowHint")
        if gas_flow_hint.count() > 0:
            logger.info("Found #gasFlowHint")
        else:
            logger.error("ERROR: #gasFlowHint not found")

        condition_locator = page.locator("#standardCondition")
        described_by = condition_locator.get_attribute("aria-describedby")
        logger.info(f"Standard Condition aria-describedby: {described_by}")

        if described_by == "gasFlowHint":
            logger.info("SUCCESS: aria-describedby matches hint ID")
        else:
            logger.error("ERROR: aria-describedby mismatch")

        browser.close()


if __name__ == "__main__":
    run()
