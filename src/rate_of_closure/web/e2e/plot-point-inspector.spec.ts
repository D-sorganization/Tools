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

const selectedMarkerPoint = async (canvas: Locator) => canvas.evaluate((element) => {
  const target = element as HTMLCanvasElement;
  const context = target.getContext("2d");
  if (context === null) throw new Error("plot canvas context is unavailable");
  const image = context.getImageData(0, 0, target.width, target.height);
  const matches: Array<[number, number]> = [];
  for (let y = 55; y < target.height - 45; y += 1) {
    for (let x = 55; x < target.width - 145; x += 1) {
      const pixel = (y * target.width + x) * 4;
      if (image.data[pixel] >= 245 && image.data[pixel + 1] >= 247 &&
          image.data[pixel + 2] >= 249 && image.data[pixel + 3] === 255) {
        matches.push([x, y]);
      }
    }
  }
  if (matches.length < 8) throw new Error("selected plot marker was not rendered");
  const center = matches.reduce(([x, y], point) => [x + point[0], y + point[1]], [0, 0]);
  return {
    x: center[0] / matches.length,
    y: center[1] / matches.length,
    width: target.width,
    height: target.height,
  };
});

for (const viewport of [
  { width: 1440, height: 900 },
  { width: 1280, height: 720 },
  { width: 390, height: 844 },
]) {
  test(`plot exact-point inspector is direct at ${viewport.width}x${viewport.height}`, async (
    { page },
    testInfo,
  ) => {
    test.skip(testInfo.project.name !== "chromium-desktop", "canonical plot viewports");
    const pageErrors = capturePageErrors(page);
    await page.setViewportSize(viewport);
    await page.goto("/");
    await page.locator("#primary-tab-plots").click();
    const canvas = page.getByRole("img", { name: "Closure Sweep plot", exact: true });
    const status = page.getByRole("status").filter({ hasText: /exact point|source point/i });
    await expect(canvas).toBeVisible();
    const minimumWidth = viewport.width < 1280 ? 120 : 240;
    const minimumHeight = viewport.width < 1280 ? 180 : 240;
    await expect.poll(async () => (await visibleIntersection(canvas)).width)
      .toBeGreaterThanOrEqual(minimumWidth);
    await expect.poll(async () => (await visibleIntersection(canvas)).height)
      .toBeGreaterThanOrEqual(minimumHeight);
    expect(await page.evaluate(() => Math.max(
      document.body.scrollWidth - document.body.clientWidth,
      document.documentElement.scrollWidth - document.documentElement.clientWidth,
    ))).toBe(0);

    await canvas.focus();
    await canvas.press("Home");
    await expect(status).toContainText("source point 1/");
    const marker = await selectedMarkerPoint(canvas);
    await canvas.press("End");
    await expect(status).not.toContainText("source point 1/");
    const box = await canvas.boundingBox();
    if (box === null) throw new Error("plot canvas has no rendered rectangle");
    await page.mouse.click(
      box.x + marker.x * box.width / marker.width,
      box.y + marker.y * box.height / marker.height,
    );
    await expect(status).toContainText("source point 1/");
    await expect(canvas).toBeFocused();
    expect((await visibleIntersection(status)).height).toBeGreaterThan(0);
    await testInfo.attach(`plot-selected-${viewport.width}x${viewport.height}`, {
      body: await page.screenshot(),
      contentType: "image/png",
    });
    expect(pageErrors).toEqual([]);
  });
}
