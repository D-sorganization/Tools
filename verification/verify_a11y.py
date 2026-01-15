
from playwright.sync_api import sync_playwright

def verify_accessibility_attributes():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the page directly from file
        import os
        cwd = os.getcwd()
        page.goto(f'file://{cwd}/web_applications/unit_converter/unit-converter-app/index.html')

        # Verify Conversion Factor hint association
        print('Verifying Conversion Factor...')
        input_elem = page.locator('#conversionFactor')
        hint_id = input_elem.get_attribute('aria-describedby')
        assert 'conversionFactorHint' in hint_id
        hint_elem = page.locator(f'#{hint_id}')
        assert hint_elem.count() == 1
        print('✅ Conversion Factor associated with hint')

        # Verify Custom Aliases hint association
        print('Verifying Custom Aliases...')
        input_elem = page.locator('#customAliases')
        hint_id = input_elem.get_attribute('aria-describedby')
        assert 'customAliasesHint' in hint_id
        hint_elem = page.locator(f'#{hint_id}')
        assert hint_elem.count() == 1
        print('✅ Custom Aliases associated with hint')

        # Verify Gas Density hint association
        print('Verifying Gas Density...')
        input_elem = page.locator('#gasDensity')
        hint_ids = input_elem.get_attribute('aria-describedby').split()
        assert 'gasDensityHint' in hint_ids
        assert 'heatingValHint' in hint_ids
        print('✅ Gas Density associated with both hints')

        # Take a screenshot for visual confirmation of the page load
        page.screenshot(path='verification/accessibility_check.png')
        browser.close()

if __name__ == '__main__':
    verify_accessibility_attributes()
