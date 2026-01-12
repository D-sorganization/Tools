from playwright.sync_api import sync_playwright, expect

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto("http://localhost:5000")
            expect(page).to_have_title("Aurora CAS Calculator")

            # Take a screenshot of the initial state
            page.screenshot(path="verification/initial_state.png")
            print("Initial state screenshot taken.")

            # Try to click the enter button
            page.get_by_role("button", name="ENTER").click()

            # Take a screenshot after click (it might be too fast to catch loading state without artificial delay in backend, but let's see)
            page.screenshot(path="verification/after_click.png")
            print("After click screenshot taken.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    run()
