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
      const left = Math.max(rect.left, clip.left);
      const top = Math.max(rect.top, clip.top);
      rect = new DOMRect(
        left,
        top,
        Math.max(0, Math.min(rect.right, clip.right) - left),
        Math.max(0, Math.min(rect.bottom, clip.bottom) - top),
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
  test(`simulation scrub authority is exact at ${viewport.width}x${viewport.height}`, async (
    { page },
    testInfo,
  ) => {
    test.skip(testInfo.project.name !== "chromium-desktop", "canonical simulation viewports");
    const pageErrors = capturePageErrors(page);
    await page.setViewportSize(viewport);
    await page.goto("/");
    await page.locator("#primary-tab-simulation").click();

    const scene = page.getByLabel(
      "Simulation scene with selectable screw-axis motion glyph",
    );
    const setup = page.getByRole("region", { name: "Simulation setup" });
    await expect(scene).toBeVisible();
    await expect(setup).toBeVisible();
    const minimumWidth = viewport.width < 1280 ? 120 : 240;
    const minimumHeight = viewport.width < 1280 ? 180 : 240;
    await expect.poll(async () => (await visibleIntersection(scene)).width)
      .toBeGreaterThanOrEqual(minimumWidth);
    await expect.poll(async () => (await visibleIntersection(scene)).height)
      .toBeGreaterThanOrEqual(minimumHeight);
    const sceneRect = await scene.boundingBox();
    const setupRect = await setup.boundingBox();
    if (sceneRect === null || setupRect === null) throw new Error("simulation geometry unavailable");
    if (viewport.width < 1280) expect(sceneRect.y).toBeLessThanOrEqual(setupRect.y);
    expect(await page.evaluate(() => Math.max(
      document.body.scrollWidth - document.body.clientWidth,
      document.documentElement.scrollWidth - document.documentElement.clientWidth,
    ))).toBe(0);
    await testInfo.attach(`simulation-result-${viewport.width}x${viewport.height}`, {
      body: await page.screenshot(),
      contentType: "image/png",
    });

    const slider = page.getByRole("slider", { name: "Impact Time" });
    await slider.focus();
    const before = Number(await slider.inputValue());
    await slider.press("ArrowLeft");
    await expect(slider).toHaveValue(String(before - 1));
    await expect(page.getByText("Completed — impact and flight available"))
      .toBeVisible();
    await page.getByRole("button", { name: "Auto τ" }).click();
    await expect(page.getByText("auto", { exact: true })).toBeVisible();
    await expect(page.getByText("Completed — impact and flight available"))
      .toBeVisible();

    await page.getByLabel("Swing Source").selectOption("double_pendulum");
    await expect(page.getByText("Inputs changed — run required")).toBeVisible();
    await testInfo.attach(`simulation-stale-${viewport.width}x${viewport.height}`, {
      body: await page.screenshot({ fullPage: true }),
      contentType: "image/png",
    });
    expect(pageErrors).toEqual([]);
  });
}
