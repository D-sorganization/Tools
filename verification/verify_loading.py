from playwright.sync_api import sync_playwright, expect
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        try:
            page.goto("http://localhost:5000")

            page.locator("input#expression").fill("1+1")

            # Robust fetch override
            page.evaluate("""
                const originalFetch = window.fetch;
                window.fetch = async (url, options) => {
                    // Check if url is string and includes endpoint
                    if (typeof url === 'string' && url.includes('/api/calculate')) {
                        await new Promise(r => setTimeout(r, 2000));
                    }
                    return originalFetch(url, options);
                };
            """)

            # Click ENTER
            execute_btn = page.locator("#execute")
            execute_btn.click()

            # Check if button text changed to "Processing..."
            expect(execute_btn).to_have_text("Processing...", timeout=1000)
            expect(execute_btn).to_be_disabled()

            page.screenshot(path="verification/loading_state.png")
            print("Loading state verified and screenshot taken.")

            # Wait for restoration
            expect(execute_btn).to_have_text("ENTER", timeout=5000)
            expect(execute_btn).to_be_enabled()

            print("Button restored state verified.")

            # 2. Verify required field validation
            page.locator("input#expression").fill("")
            page.locator("input#variable").click()
            page.screenshot(path="verification/invalid_state.png")
            print("Invalid state screenshot taken.")

        except Exception as e:
            # Capture what happened if it failed
            page.screenshot(path="verification/failure.png")
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    run()
