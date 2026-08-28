import { expect, test, type Locator } from "@playwright/test";

import { capturePageErrors } from "./variationTestSupport";

const visibleIntersection = async (locator: Locator) => locator.evaluate((element) => {
  let rect = element.getBoundingClientRect();
  let ancestor = element.parentElement;
  while (ancestor !== null) {
    const style = getComputedStyle(ancestor);
    if ([style.overflow, style.overflowX, style.overflowY]
      .some((value) => ["hidden", "clip", "scroll", "auto"].includes(value))) {
      const clip = ancestor.getBoundingClientRect();
      const left = Math.max(rect.left, clip.left); const top = Math.max(rect.top, clip.top);
      const right = Math.min(rect.right, clip.right); const bottom = Math.min(rect.bottom, clip.bottom);
      rect = new DOMRect(left, top, Math.max(0, right - left), Math.max(0, bottom - top));
    }
    ancestor = ancestor.parentElement;
  }
  const left = Math.max(rect.left, 0); const top = Math.max(rect.top, 0);
  const right = Math.min(rect.right, innerWidth); const bottom = Math.min(rect.bottom, innerHeight);
  return { width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
});

for (const viewport of [
  { width: 1440, height: 900 },
  { width: 1280, height: 720 },
  { width: 390, height: 844 },
]) {
  test(`putting sample inspector is synchronized at ${viewport.width}x${viewport.height}`, async (
    { page }, testInfo,
  ) => {
    test.skip(testInfo.project.name !== "chromium-desktop", "canonical putting viewports");
    const pageErrors = capturePageErrors(page);
    await page.setViewportSize(viewport);
    await page.goto("/");
    await page.locator("#primary-tab-putting").click();
    const path = page.getByRole("img", { name: "Interactive putt path sample inspector" });
    const speed = page.getByRole("img", { name: "Interactive synchronized speed versus distance plot" });
    await expect(path).toBeVisible();
    await expect(speed).toBeVisible();
    const intersection = await visibleIntersection(path);
    expect(intersection.width).toBeGreaterThanOrEqual(viewport.width < 1280 ? 120 : 240);
    expect(intersection.height).toBeGreaterThanOrEqual(viewport.width < 1280 ? 180 : 240);
    expect(await page.evaluate(() => Math.max(
      document.body.scrollWidth - document.body.clientWidth,
      document.documentElement.scrollWidth - document.documentElement.clientWidth,
    ))).toBe(0);

    const start = path.locator("circle[fill='#f8fafc']");
    const startBox = await start.boundingBox();
    if (startBox === null) throw new Error("putting start sample has no rendered rectangle");
    await page.mouse.click(startBox.x + startBox.width / 2, startBox.y + startBox.height / 2);
    const status = page.getByRole("status", { name: "Selected putt sample" });
    await expect(status).toContainText("Source sample 0 (zero-based)");
    await expect(page.getByTestId("putting-selected-sample")).toHaveCount(2);
    const displayedContext = page.getByLabel("Displayed putting result context");
    const acceptedContext = await displayedContext.textContent();
    await path.focus();
    await path.press("End");
    await expect(status).toContainText(/Source sample \d+ \(zero-based\).*pure roll/);
    await path.press("Escape");
    await expect(status).toHaveText("No trajectory sample selected.");
    await path.press("Home");
    await expect(status).toContainText("Source sample 0 (zero-based)");
    await testInfo.attach(`putting-selected-${viewport.width}x${viewport.height}`, {
      body: await page.screenshot(), contentType: "image/png",
    });
    await expect(displayedContext).toHaveText(acceptedContext ?? "");
    expect(pageErrors).toEqual([]);
  });
}
