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
      rect = new DOMRect(
        Math.max(rect.left, clip.left),
        Math.max(rect.top, clip.top),
        Math.max(0, Math.min(rect.right, clip.right) - Math.max(rect.left, clip.left)),
        Math.max(0, Math.min(rect.bottom, clip.bottom) - Math.max(rect.top, clip.top)),
      );
    }
    ancestor = ancestor.parentElement;
  }
  return {
    width: Math.max(0, Math.min(rect.right, innerWidth) - Math.max(rect.left, 0)),
    height: Math.max(0, Math.min(rect.bottom, innerHeight) - Math.max(rect.top, 0)),
  };
});

for (const viewport of [
  { width: 1440, height: 900 },
  { width: 1280, height: 720 },
  { width: 390, height: 844 },
]) {
  test(`flight sample inspector is synchronized at ${viewport.width}x${viewport.height}`, async (
    { page },
    testInfo,
  ) => {
    test.skip(testInfo.project.name !== "chromium-desktop", "canonical flight viewports");
    const pageErrors = capturePageErrors(page);
    await page.setViewportSize(viewport);
    await page.goto("/");
    await page.locator("#primary-tab-flight").click();
    await page.getByRole("checkbox", {
      name: "Compare No Wind and Selected Wind",
    }).click();
    await page.getByRole("button", { name: "Run Flight" }).click();
    const side = page.getByLabel("Flight side profile (height vs carry)");
    const top = page.getByLabel("Flight top-down view (lateral vs carry)");
    await expect(side).toBeVisible();
    await expect(top).toBeVisible();
    const minimumWidth = viewport.width < 1280 ? 120 : 240;
    const minimumHeight = viewport.width < 1280 ? 180 : 240;
    await expect.poll(async () => (await visibleIntersection(side)).width)
      .toBeGreaterThanOrEqual(minimumWidth);
    await expect.poll(async () => (await visibleIntersection(side)).height)
      .toBeGreaterThanOrEqual(minimumHeight);
    expect(await page.evaluate(() => Math.max(
      document.body.scrollWidth - document.body.clientWidth,
      document.documentElement.scrollWidth - document.documentElement.clientWidth,
    ))).toBe(0);

    const context = page.getByRole("status", { name: "Displayed flight context" });
    const acceptedContext = await context.textContent();
    const status = page.getByRole("status", { name: "Selected flight sample" });
    const playback = page.getByLabel("Ball flight playback position");
    const sideBox = await side.boundingBox();
    if (sideBox === null) throw new Error("flight side profile has no rendered rectangle");
    await page.mouse.click(
      sideBox.x + sideBox.width * (34 / 860),
      sideBox.y + sideBox.height * (226 / 260),
    );
    await expect(status).toContainText("source sample 1/");
    await expect(status).toContainText("launch");
    await expect(playback).toContainText(/^0\.00/);
    expect((await visibleIntersection(status)).height).toBeGreaterThan(0);
    expect((await visibleIntersection(playback)).height).toBeGreaterThan(0);
    await side.press("End");
    await expect(status).toContainText("landing");
    await side.press("Home");
    await expect(status).toContainText("source sample 1/");
    await expect(side).toBeFocused();
    await expect(context).toHaveText(acceptedContext ?? "");
    await testInfo.attach(`flight-selected-${viewport.width}x${viewport.height}`, {
      body: await page.screenshot(),
      contentType: "image/png",
    });
    expect(pageErrors).toEqual([]);
  });
}
