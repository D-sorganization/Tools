import os

from playwright.sync_api import sync_playwright


def run() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Load the local HTML file
        # We need absolute path
        cwd = os.getcwd()
        file_path = f"file://{cwd}/src/web_applications/unit_converter/unit-converter-app/index.html"
        print(f"Loading: {file_path}")
        page.goto(file_path)

        # 1. Verify Keyboard Shortcut Hint
        print("Verifying Keyboard Shortcut Hint...")
        # Check if the "From" label contains the shortcut hint
        from_label = page.locator("label[for='fromValue']")
        # Screenshot the label area
        from_label.screenshot(path="verification/shortcut_hint.png")
        print("Screenshot saved to verification/shortcut_hint.png")

        # 2. Verify Accessibility Attributes
        print("Verifying Accessibility Attributes...")

        # Check gas flow hint ID
        # There are multiple .param-hint, we want the one for gas flow
        gas_flow_hint = page.locator("#gasFlowHint")
        if gas_flow_hint.count() > 0:
            print("Found #gasFlowHint")
        else:
            print("ERROR: #gasFlowHint not found")

        # Check aria-describedby on standardCondition
        described_by = page.locator("#standardCondition").get_attribute(
            "aria-describedby"
        )
        print(f"Standard Condition aria-describedby: {described_by}")

        if described_by == "gasFlowHint":
            print("SUCCESS: aria-describedby matches hint ID")
        else:
            print("ERROR: aria-describedby mismatch")

        browser.close()


if __name__ == "__main__":
    run()
