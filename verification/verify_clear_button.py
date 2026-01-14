
from playwright.sync_api import sync_playwright, expect
import os

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    # Load the local HTML file
    cwd = os.getcwd()
    file_path = f"file://{cwd}/web_applications/unit_converter/unit-converter-app/index.html"
    page.goto(file_path)

    # Check initial state: input has value '1' (set by init), button should be visible
    from_input = page.locator("#fromValue")
    clear_button = page.locator("#clearInput")

    expect(from_input).to_have_value("1")
    expect(clear_button).to_be_visible()

    print("Initial state verified: Input has value '1', button is visible.")

    # Take screenshot of visible button
    page.screenshot(path="verification/clear_button_visible.png")

    # Clear the input using the button
    clear_button.click()

    # Verify input is empty and button is hidden
    expect(from_input).to_have_value("")
    expect(clear_button).not_to_be_visible()

    print("Clear action verified: Input is empty, button is hidden.")

    # Type new value
    from_input.fill("123")

    # Verify button reappears
    expect(clear_button).to_be_visible()

    print("Re-entry verified: Button reappears on typing.")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
