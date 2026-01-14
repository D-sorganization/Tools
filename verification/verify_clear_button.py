
from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        app_path = os.path.abspath('web_applications/unit_converter/unit-converter-app/index.html')
        page.goto(f'file://{app_path}')

        # Wait for app to initialize (value should be '1')
        page.wait_for_selector('#fromValue')

        # Verify initial state: Clear button should be visible because default value is '1'
        print('Verifying initial state...')
        clear_btn = page.locator('#clearInput')
        if clear_btn.is_visible():
            print('Clear button visible initially (Correct)')
        else:
            print('Clear button NOT visible initially (Incorrect)')

        # Take screenshot of initial state
        page.screenshot(path='verification/1_initial_state.png')

        # Type something
        print('Typing into input...')
        page.fill('#fromValue', '123')

        # Take screenshot showing button
        page.screenshot(path='verification/2_with_input.png')

        # Click clear button
        print('Clicking clear button...')
        clear_btn.click()

        # Verify input is cleared
        val = page.input_value('#fromValue')
        print(f'Input value after clear: "{val}"')
        if val == '':
            print('Input cleared successfully')
        else:
            print('Input NOT cleared')

        # Verify button is hidden
        if not clear_btn.is_visible():
             print('Clear button hidden after clearing (Correct)')
        else:
             print('Clear button visible after clearing (Incorrect)')

        # Take screenshot of cleared state
        page.screenshot(path='verification/3_cleared.png')

        browser.close()

if __name__ == '__main__':
    run()
