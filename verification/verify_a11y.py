
from playwright.sync_api import sync_playwright

def verify_accessibility_attributes():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Loading the local file directly
        page.goto('file:///app/web_applications/unit_converter/unit-converter-app/index.html')

        # Open the modal first so elements are technically 'visible' to playwright if it checks display
        page.locator('#customUnitsButton').click()

        # Verify Conversion Factor Hint association
        conversion_input = page.locator('#conversionFactor')
        hint_id = conversion_input.get_attribute('aria-describedby')
        assert hint_id == 'conversionFactorHint', f'Expected conversionFactorHint, got {hint_id}'

        # We don't strictly need is_visible() for accessibility attributes verification if the modal logic is complex
        # but let's check if the ID exists in the page
        hint_element = page.locator(f'#{hint_id}')
        # Just check count
        assert hint_element.count() == 1, 'Hint element not found in DOM'
        print('✅ Conversion Factor Hint associated correctly')

        # Verify Custom Aliases Hint association
        aliases_input = page.locator('#customAliases')
        aliases_hint_id = aliases_input.get_attribute('aria-describedby')
        assert aliases_hint_id == 'customAliasesHint', f'Expected customAliasesHint, got {aliases_hint_id}'
        print('✅ Custom Aliases Hint associated correctly')

        # Verify Gas Density Hint association
        # This one is in the main page, but hidden initially.
        gas_density_input = page.locator('#gasDensity')
        gas_density_hint_ids = gas_density_input.get_attribute('aria-describedby')
        assert 'heatingValHint' in gas_density_hint_ids and 'gasDensityHint' in gas_density_hint_ids, f'Expected both hints, got {gas_density_hint_ids}'
        print('✅ Gas Density Hint associated correctly')

        page.screenshot(path='verification/accessibility_check.png')

        browser.close()

if __name__ == '__main__':
    try:
        verify_accessibility_attributes()
        print('Verification successful!')
    except Exception as e:
        print(f'Verification failed: {e}')
        exit(1)
